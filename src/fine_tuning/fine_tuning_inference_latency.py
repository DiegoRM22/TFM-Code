"""
Fine-tuning inference con medición de latencia por pregunta (H4).
Ejecuta sobre 5 preguntas del test set. Output incluye latency_s.
"""
import time
import torch
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

checkpoint_dir = MODELS_DIR / "fine_tuning" / "colab_checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {checkpoint_dir}")
adapter_path = str(checkpoints[-1])
print(f"Usando checkpoint: {adapter_path}")

test_dataset_path = DATA_DIR / "shared" / "dataset_test.json"
output_path       = DATA_DIR / "fine_tuning" / "fine_tuning_latency_results.json"

N_SAMPLES = 5

print("1. Cargando configuración técnica...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

print("2. Cargando modelo base...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

print("3. Cargando adaptador LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

test_data = []
print("4. Parseando dataset de test...")
with open(test_dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        full_text = obj.get('text', '')

        contexto_match = re.search(r'Contexto:\s*(.*?)\n\nPregunta:', full_text, re.DOTALL)
        pregunta_match = re.search(r'Pregunta:\s*(.*?)<\|eot_id\|>', full_text)
        gt_match = re.search(r'assistant<\|end_header_id\|>\n(.*?)(?:<\|eot_id\|>|$)', full_text, re.DOTALL)

        if pregunta_match and gt_match:
            test_data.append({
                "question":     pregunta_match.group(1).strip(),
                "ground_truth": gt_match.group(1).strip(),
                "context":      contexto_match.group(1).strip() if contexto_match else "",
            })

test_data = test_data[:N_SAMPLES]
print(f"Muestras a procesar: {len(test_data)}")

results = []
print(f"\n5. Inferencia fine-tuning con latencia ({N_SAMPLES} preguntas)...")
for i, entry in enumerate(test_data):
    pregunta = entry['question']
    contexto = entry['context']

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"Eres un asistente legal experto en el Reglamento General de Protección de Datos (RGPD) "
        f"de la Unión Europea. Responde a la pregunta basándote ÚNICAMENTE en el contexto proporcionado. "
        f"Sé directo y cita el artículo exacto si aparece en el contexto."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Contexto: {contexto}\n\nPregunta: {pregunta}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    t_start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("assistant")[-1].strip()
    latency_s = round(time.perf_counter() - t_start, 3)

    results.append({
        "question":     pregunta,
        "answer":       answer,
        "contexts":     [contexto] if contexto else [],
        "ground_truth": entry['ground_truth'],
        "latency_s":    latency_s,
    })
    print(f"[{i+1}/{len(test_data)}] Procesada — {latency_s}s")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

latencias = [r["latency_s"] for r in results]
print(f"\nResultados guardados en: {output_path}")
print(f"Latencia media: {sum(latencias)/len(latencias):.2f}s | min: {min(latencias):.2f}s | max: {max(latencias):.2f}s")
