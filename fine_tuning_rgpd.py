import json
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# ==========================================
# 1. CARGA Y PREPARACIÓN DEL DATASET
# ==========================================
print("1. Cargando y formateando el dataset...")

archivo_json = "golden_dataset_raw_limpio100.json" 

with open(archivo_json, 'r', encoding='utf-8') as f:
    datos_brutos = json.load(f)

# Función para formatear al estilo "Instruct" que Llama-3 entiende
def formatear_prompt(ejemplo):
    contexto = ejemplo['contexts'][0]
    pregunta = ejemplo['question']
    respuesta = ejemplo['ground_truth']
    
    # Prompt estructurado que fuerza al modelo a basarse en el contexto
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un asistente legal experto en el Reglamento General de Protección de Datos (RGPD) de la Unión Europea. Responde a la pregunta basándote ÚNICAMENTE en el contexto proporcionado.<|eot_id|><|start_header_id|>user<|end_header_id|>
Contexto: {contexto}

Pregunta: {pregunta}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{respuesta}<|eot_id|>"""
    
    return {"text": prompt}

datos_formateados = [formatear_prompt(d) for d in datos_brutos]

# ==========================================
# 2. SPLIT: TRAIN (70%) - VAL (15%) - TEST (15%)
# ==========================================
# Primero separamos el 30% para Val+Test
train_data, temp_data = train_test_split(datos_formateados, test_size=0.30, random_state=42)
# Luego dividimos ese 30% a la mitad (15% y 15%)
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

# Convertir a formato Dataset de HuggingFace
dataset = DatasetDict({
    'train': Dataset.from_list(train_data),
    'validation': Dataset.from_list(val_data),
    'test': Dataset.from_list(test_data)
})

print(f"Dataset dividido: {len(dataset['train'])} Train | {len(dataset['validation'])} Val | {len(dataset['test'])} Test")

# Guardar el Test Set para evaluar después
dataset['test'].to_json("dataset_test.json", force_ascii=False)

# ==========================================
# 3. CONFIGURACIÓN DEL MODELO (QLoRA 4-bit)
# ==========================================
print("2. Cargando modelo Llama-3 en 4 bits...")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Cuantización a 4 bits para que quepa en GPU local
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Llama-3 necesita que se defina un pad_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # Asigna automáticamente a la GPU
)

model = prepare_model_for_kbit_training(model)

# Configuración de LoRA (Los "Adaptadores")
peft_config = LoraConfig(
    r=16, # Rango (16 es un buen equilibrio entre precisión y memoria)
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Capas de atención de Llama-3
)

# model = get_peft_model(model, peft_config)

from trl import SFTConfig, SFTTrainer

# ==========================================
# 4. CONFIGURACIÓN DEL ENTRENAMIENTO
# ==========================================
print("3. Iniciando Fine-Tuning...")

sft_config = SFTConfig(
    output_dir="./resultados_rgpd_llama3",
    max_steps=200,                # Número de pasos de entrenamiento
    per_device_train_batch_size=1, # Bajamos a 1 para asegurar que no sature la VRAM
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    max_length=1024,
    dataset_text_field="text",
    packing=False,
    report_to="none"              # Evita errores si no tienes WandB instalado
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

trainer.train()

# ==========================================
# 5. GUARDAR EL MODELO FINAL
# ==========================================
print("4. Guardando adaptadores LoRA...")
trainer.model.save_pretrained("./llama3_rgpd_lora")
tokenizer.save_pretrained("./llama3_rgpd_lora")
print("¡Fine-Tuning completado con éxito!")