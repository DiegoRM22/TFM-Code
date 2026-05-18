import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_path = "./resultados_rgpd_llama3/checkpoint-200"
test_dataset_path = "dataset_test.json" 

print("1. Cargando configuración técnica (Modo Manual)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)

print("2. Cargando modelo base (Saltando orquestador Accelerate)...")
# IMPORTANTE: No usamos device_map="auto" para evitar el crash de _is_hf_initialized
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    trust_remote_code=True
)

print("3. Cargando adaptador LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)

# Movemos el modelo manualmente a la GPU (CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4. Cargar y parsear el dataset JSONL
test_data = []
print("4. Parseando dataset de test...")
with open(test_dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            full_text = obj.get('text', '')
            
            # Extraemos la pregunta y la respuesta ideal
            pregunta_match = re.search(r'Pregunta: (.*?)<\|eot_id\|>', full_text)
            gt_match = re.search(r'assistant<\|end_header_id\|>\n(.*?)(?:<\|eot_id\|>|$)', full_text, re.DOTALL)
            
            if pregunta_match and gt_match:
                test_data.append({
                    "question": pregunta_match.group(1).strip(),
                    "ground_truth": gt_match.group(1).strip()
                })

print(f"Total de muestras para evaluar: {len(test_data)}")

results = []

print("\n5. Iniciando inferencia por lotes...")
for i, entry in enumerate(test_data):
    pregunta = entry['question']
    
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEres un asistente legal experto en el RGPD.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{pregunta}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraemos solo la parte del asistente
    answer = full_response.split("assistant")[-1].strip()
    
    results.append({
        "question": pregunta,
        "answer": answer,
        "ground_truth": entry['ground_truth']
    })
    print(f"[{i+1}/{len(test_data)}] Procesada")

# 6. Guardar resultados para RAGAS
with open("resultados_para_ragas.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\n¡Evaluación completada con éxito!")