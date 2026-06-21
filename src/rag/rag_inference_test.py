"""
Inferencia RAG sobre el test set (14q) para comparación directa con FT, baseline e híbrido.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "shared" / "dataset_test.json"
output_file = DATA_DIR / "rag" / "rag_inference_test_results.json"
db_dir      = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

embeddings = BgeM3Embeddings("BAAI/bge-m3")
llm        = OllamaLLM(model="llama3:8b", temperature=0.0)

vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

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


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Parsear test set (JSONL con prompts formateados)
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
print(f"Iniciando inferencia RAG sobre {len(test_data)} preguntas del test set (k=5)...")

for i, entrada in enumerate(test_data):
    pregunta = entrada["question"]
    print(f"[{i+1}/{len(test_data)}] Procesando...", flush=True)

    docs = retriever.invoke(pregunta)
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
