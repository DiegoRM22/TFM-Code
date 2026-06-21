"""
RAG avanzado inference con medición de latencia por pregunta (H4).
Ejecuta sobre 5 preguntas. Output incluye latency_s.
Latencia incluye: HyDE expand + retrieval + generación (pipeline completo).
"""
import time
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.retriever import build_advanced_retriever, hyde_expand
from langchain_ollama import OllamaLLM

DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "shared" / "dataset_test.json"
output_file = DATA_DIR / "rag" / "rag_advanced_latency_results.json"
db_dir      = str(DATA_DIR / "vectorstore" / "db_rgpd_semantic")

N_SAMPLES = 5

TEMPLATE = """Eres un consultor jurídico especializado en el RGPD. Responde de forma DIRECTA y CONCISA.
REGLAS:
- Usa ÚNICAMENTE la información del contexto proporcionado.
- Si el artículo exacto aparece en el contexto, cítalo.
- Si la respuesta no está en el contexto, responde: "El contexto no contiene información suficiente para responder esta pregunta."
- No añadas información que no esté en el contexto.
- Responde siempre en español.

Contexto legal del RGPD:
{context}

Pregunta: {question}

Respuesta directa (máximo 3-4 frases):"""

print("1. Construyendo retriever avanzado (BM25 + Chroma + CrossEncoder)...")
retriever = build_advanced_retriever(db_dir)

print("2. Cargando LLM (llama3:8b)...")
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

test_data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        full_text = obj.get("text", "")
        pregunta_match = re.search(r'Pregunta:\s*(.*?)<\|eot_id\|>', full_text)
        gt_match = re.search(r'assistant<\|end_header_id\|>\n(.*?)(?:<\|eot_id\|>|$)', full_text, re.DOTALL)
        if pregunta_match and gt_match:
            test_data.append({
                "question":     pregunta_match.group(1).strip(),
                "ground_truth": gt_match.group(1).strip(),
            })

test_data = test_data[:N_SAMPLES]

results = []
print(f"3. Inferencia RAG avanzado con latencia ({N_SAMPLES} preguntas)...")

for i, entrada in enumerate(test_data):
    pregunta = entrada["question"]
    print(f"[{i+1}/{len(test_data)}] Procesando...", flush=True)

    # Latencia incluye HyDE + retrieval + generación
    t_start = time.perf_counter()
    query_expandida = hyde_expand(pregunta, llm)
    docs = retriever.invoke(query_expandida)
    contextos = [doc.page_content for doc in docs]
    contexto_str = "\n\n".join(contextos)
    respuesta = llm.invoke(TEMPLATE.format(context=contexto_str, question=pregunta))
    latency_s = round(time.perf_counter() - t_start, 3)

    results.append({
        "question":     pregunta,
        "answer":       respuesta,
        "contexts":     contextos,
        "ground_truth": entrada["ground_truth"],
        "latency_s":    latency_s,
    })
    print(f"  -> {latency_s}s")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

latencias = [r["latency_s"] for r in results]
print(f"\nDataset generado en '{output_file}' ({len(results)} entradas)")
print(f"Latencia media: {sum(latencias)/len(latencias):.2f}s | min: {min(latencias):.2f}s | max: {max(latencias):.2f}s")
