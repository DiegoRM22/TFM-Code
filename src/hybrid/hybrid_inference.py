"""
Arquitectura híbrida: RAG retrieval (ChromaDB + bge-m3) + generador fine-tuneado (Llama-3-8B + LoRA).
Usa las mismas 92 preguntas que rag_evaluation.py → comparación directa RAG vs Híbrido.
"""
import json
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

# ── Rutas ──────────────────────────────────────────────────────────────────────
# Test set (14q) — mismo conjunto que FT y baseline para comparación directa
input_file  = DATA_DIR / "shared" / "dataset_test.json"
output_file = DATA_DIR / "hybrid" / "hybrid_inference_results.json"
db_dir      = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

checkpoint_dir = MODELS_DIR / "fine_tuning" / "colab_checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {checkpoint_dir}")
adapter_path = str(checkpoints[-1])

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# ── Retriever (bge-m3 + ChromaDB) ─────────────────────────────────────────────
print("1. Cargando retriever (bge-m3 + ChromaDB)...")
embeddings = BgeM3Embeddings("BAAI/bge-m3")
vector_db  = Chroma(persist_directory=db_dir, embedding_function=embeddings)
retriever  = vector_db.as_retriever(search_kwargs={"k": 5})

# ── Generador (Llama-3-8B + LoRA) ─────────────────────────────────────────────
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

# ── Dataset (JSONL con prompts formateados) ────────────────────────────────────
import re

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

# ── Inferencia ─────────────────────────────────────────────────────────────────
results = []
print(f"\n3. Inferencia híbrida sobre {len(golden_dataset)} preguntas (k=5)...")

for i, entrada in enumerate(golden_dataset):
    pregunta = entrada["question"]
    print(f"[{i+1}/{len(golden_dataset)}] Procesando...", flush=True)

    # Recuperar contextos con RAG (no usar contexto del training)
    docs = retriever.invoke(pregunta)
    contextos = [doc.page_content for doc in docs]
    contexto_str = "\n\n".join(contextos)

    # Generar con modelo fine-tuneado
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

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("assistant")[-1].strip()

    results.append({
        "question":     pregunta,
        "answer":       answer,
        "contexts":     contextos,
        "ground_truth": entrada["ground_truth"],
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nResultados guardados en: {output_file} ({len(results)} entradas)")
