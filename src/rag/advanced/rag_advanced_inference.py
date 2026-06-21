"""
RAG avanzado (S2b): EnsembleRetriever (BM25+Chroma) + CrossEncoder reranking + HyDE.
Vectorstore semántico (db_rgpd_semantic). Test set 14q para comparación directa.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.retriever import build_advanced_retriever, hyde_expand
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "shared" / "dataset_test.json"
output_file = DATA_DIR / "rag" / "advanced" / "rag_advanced_inference_dataset.json"
db_dir      = str(DATA_DIR / "vectorstore" / "db_rgpd_semantic")

template = """Eres un consultor jurídico especializado en el RGPD. Responde de forma DIRECTA y CONCISA.
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

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


print("1. Construyendo retriever avanzado (BM25 + Chroma + CrossEncoder)...")
retriever = build_advanced_retriever(db_dir)

print("2. Cargando LLM (llama3:8b)...")
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

rag_chain = (
    {"context": lambda q: format_docs(retriever.invoke(q)), "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

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

results = []
print(f"3. Inferencia RAG avanzado sobre {len(test_data)} preguntas (HyDE + BM25+Chroma + reranking)...")

for i, entrada in enumerate(test_data):
    pregunta = entrada["question"]
    print(f"[{i+1}/{len(test_data)}] Procesando...", flush=True)

    query_expandida = hyde_expand(pregunta, llm)
    docs = retriever.invoke(query_expandida)
    contextos = [doc.page_content for doc in docs]
    respuesta = rag_chain.invoke(pregunta)

    results.append({
        "question":     pregunta,
        "answer":       respuesta,
        "contexts":     contextos,
        "ground_truth": entrada["ground_truth"],
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nDataset generado en '{output_file}' ({len(results)} entradas)")
