"""
Híbrido avanzado no-semántico: EnsembleRetriever (BM25+Chroma) + CrossEncoder reranking
+ generador fine-tuneado (Llama-3-8B + LoRA). Sin HyDE. Vectorstore pymupdf.
Test set 14q — comparación directa con versión semántica (v3).
"""
import json
import re
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.retriever import build_advanced_retriever

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

input_file  = DATA_DIR / "shared" / "dataset_test.json"
output_file = DATA_DIR / "hybrid" / "advanced" / "no-semantic" / "hybrid_advanced_no_semantic_inference_results.json"
db_dir      = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

checkpoint_dir = MODELS_DIR / "fine_tuning" / "colab_checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {checkpoint_dir}")
adapter_path = str(checkpoints[-1])

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print("1. Construyendo retriever avanzado no-semantico (BM25 + Chroma pymupdf + CrossEncoder)...")
retriever = build_advanced_retriever(db_dir)

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

results = []
print(f"\n3. Inferencia hibrida avanzada no-semantica sobre {len(golden_dataset)} preguntas...")

for i, entrada in enumerate(golden_dataset):
    pregunta = entrada["question"]
    print(f"[{i+1}/{len(golden_dataset)}] Procesando...", flush=True)

    docs = retriever.invoke(pregunta)
    contextos = [doc.page_content for doc in docs]
    contexto_str = "\n\n".join(contextos)

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"Eres un asistente legal experto en el Reglamento General de Proteccion de Datos (RGPD) "
        f"de la Union Europea. Responde a la pregunta basandote UNICAMENTE en el contexto proporcionado. "
        f"Se directo y cita el articulo exacto si aparece en el contexto."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Contexto: {contexto_str}\n\nPregunta: {pregunta}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

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

    results.append({
        "question":     pregunta,
        "answer":       answer,
        "contexts":     contextos,
        "ground_truth": entrada["ground_truth"],
    })
    print(f"  -> OK", flush=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nResultados guardados en: {output_file} ({len(results)} entradas)")
