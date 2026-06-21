"""
RAG inference con medición de latencia por pregunta (H4).
Ejecuta sobre 5 preguntas. Output incluye latency_s.
Nota: retrieval + generación se miden juntos (latencia end-to-end por pregunta).
Retriever se llama una sola vez (sin doble llamada del chain original).
"""
import time
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

DATA_DIR = PROJECT_ROOT / "data"

archivo_input  = DATA_DIR / "shared" / "golden_dataset_raw_limpio100.json"
archivo_output = DATA_DIR / "rag" / "rag_latency_results.json"
directorio_db  = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

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

embeddings = BgeM3Embeddings("BAAI/bge-m3")
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

vector_db = Chroma(persist_directory=directorio_db, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

with open(str(archivo_input), 'r', encoding='utf-8') as f:
    golden_dataset = json.load(f)

golden_dataset = golden_dataset[:N_SAMPLES]

results = []
print(f"Iniciando inferencia RAG con latencia ({N_SAMPLES} preguntas)...")

for i, entrada in enumerate(golden_dataset):
    pregunta = entrada['question']
    print(f"[{i+1}/{len(golden_dataset)}] Procesando...", flush=True)

    t_start = time.perf_counter()
    docs = retriever.invoke(pregunta)
    contextos_recuperados = [doc.page_content for doc in docs]
    contexto_str = "\n\n".join(contextos_recuperados)
    respuesta = llm.invoke(TEMPLATE.format(context=contexto_str, question=pregunta))
    latency_s = round(time.perf_counter() - t_start, 3)

    results.append({
        "question":     pregunta,
        "answer":       respuesta,
        "contexts":     contextos_recuperados,
        "ground_truth": entrada['ground_truth'],
        "latency_s":    latency_s,
    })
    print(f"  -> {latency_s}s")

with open(str(archivo_output), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

latencias = [r["latency_s"] for r in results]
print(f"\nDataset generado en '{archivo_output}' ({len(results)} entradas)")
print(f"Latencia media: {sum(latencias)/len(latencias):.2f}s | min: {min(latencias):.2f}s | max: {max(latencias):.2f}s")
