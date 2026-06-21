import json
import os
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

print("Cargando y formateando el dataset...")

archivo_json = DATA_DIR / "shared" / "golden_dataset_raw_limpio100.json"

with open(str(archivo_json), 'r', encoding='utf-8') as f:
    datos_brutos = json.load(f)


def formatear_prompt(ejemplo):
    contexto  = ejemplo['contexts'][0]
    pregunta  = ejemplo['question']
    respuesta = ejemplo['ground_truth']
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Eres un asistente legal experto en el Reglamento General de Protección de Datos "
        f"(RGPD) de la Unión Europea. Responde a la pregunta basándote ÚNICAMENTE en el "
        f"contexto proporcionado. Sé directo y cita el artículo exacto si aparece en el contexto."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Contexto: {contexto}\n\nPregunta: {pregunta}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        f"{respuesta}<|eot_id|>"
    )
    return {"text": prompt}


datos_formateados = [formatear_prompt(d) for d in datos_brutos]

train_data, temp_data = train_test_split(datos_formateados, test_size=0.30, random_state=42)
val_data, test_data   = train_test_split(temp_data, test_size=0.50, random_state=42)

dataset = DatasetDict({
    'train':      Dataset.from_list(train_data),
    'validation': Dataset.from_list(val_data),
    'test':       Dataset.from_list(test_data),
})

dataset['test'].to_json(str(DATA_DIR / "shared" / "dataset_test.json"), force_ascii=False)

print(f"Dataset: {len(dataset['train'])} Train | {len(dataset['validation'])} Val | {len(dataset['test'])} Test")

print("Cargando modelo Llama-3 en 4 bits...")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

output_dir = str(MODELS_DIR / "fine_tuning" / "local_checkpoints")

print("Iniciando Fine-Tuning...")
sft_config = SFTConfig(
    output_dir=output_dir,
    max_steps=120,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=10,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=30,
    save_steps=30,
    save_total_limit=2,
    load_best_model_at_end=False,
    max_length=1024,
    dataset_text_field="text",
    packing=False,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    report_to="none",
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

print("Guardando adaptadores LoRA...")
lora_dir = str(MODELS_DIR / "fine_tuning" / "local_lora")
trainer.model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print("¡Fine-Tuning completado!")
