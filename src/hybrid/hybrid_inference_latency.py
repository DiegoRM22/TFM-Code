"""
Híbrido inference con medición de latencia por pregunta (H4).
Ejecuta sobre 5 preguntas. Output incluye latency_s.
Latencia incluye: retrieval ChromaDB + generación fine-tuned.
"""
import time
import json
import re
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

input_file  = DATA_DIR / "shared" / "dataset_test.json"
output_file = DATA_DIR / "hybrid" / "hybrid_latency_results.json"
db_dir      = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

N_SAMPLES = 5

checkpoint_dir = MODELS_DIR / "fine_tuning" / "colab_checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {checkpoint_dir}")
adapter_path = str(checkpoints[-1])

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print("1. Cargando retriever (bge-m3 + ChromaDB)...")
embeddings = BgeM3Embeddings("BAAI/bge-m3")
vector_db  = Chroma(persist_directory=db_dir, embedding_function=embeddings)
retriever  = vector_db.as_retriever(search_kwargs={"k": 5})

print(f"2. Cargando generador fine-tuneado ({adapter_path})...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Generador listo en {device}.")

golden_dataset = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        full_text = obj.get("text", "")
        pregunta_match = re.search(r'Pregunta:\s*(.*?)<\|eot_id\|>', full_text)
        gt_match = re.search(r'assistant<\|end_header_id\|>\n(.*?)(?:<\|eot_id\|>|$)', full_text, re.DOTALL)
        if pregunta_match and gt_match:
            golden_dataset.append({
                "question":     pregunta_match.group(1).strip(),
                "ground_truth": gt_match.group(1).strip(),
            })

golden_dataset = golden_dataset[:N_SAMPLES]

results = []
print(f"\n3. Inferencia híbrida con latencia ({N_SAMPLES} preguntas)...")

for i, entrada in enumerate(golden_dataset):
    pregunta = entrada["question"]
    print(f"[{i+1}/{len(golden_dataset)}] Procesando...", flush=True)

    # Latencia incluye retrieval + tokenización + generación
    t_start = time.perf_counter()
    docs = retriever.invoke(pregunta)
    contextos = [doc.page_content for doc in docs]
    contexto_str = "\n\n".join(contextos)

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"Eres un asistente legal experto en el Reglamento General de Protección de Datos (RGPD) "
        f"de la Unión Europea. Responde a la pregunta basándote ÚNICAMENTE en el contexto proporcionado. "
        f"Sé directo y cita el artículo exacto si aparece en el contexto."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Contexto: {contexto_str}\n\nPregunta: {pregunta}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    latency_s = round(time.perf_counter() - t_start, 3)

    results.append({
        "question":     pregunta,
        "answer":       answer,
        "contexts":     contextos,
        "ground_truth": entrada["ground_truth"],
        "latency_s":    latency_s,
    })
    print(f"  → {latency_s}s")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

latencias = [r["latency_s"] for r in results]
print(f"\nResultados guardados en: {output_file} ({len(results)} entradas)")
print(f"Latencia media: {sum(latencias)/len(latencias):.2f}s | min: {min(latencias):.2f}s | max: {max(latencias):.2f}s")
