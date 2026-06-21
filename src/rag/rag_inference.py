import json
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

archivo_input  = DATA_DIR / "shared" / "golden_dataset_raw_limpio100.json"
archivo_output = DATA_DIR / "rag" / "rag_inference_dataset.json"
directorio_db  = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

embeddings = BgeM3Embeddings("BAAI/bge-m3")
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

vector_db = Chroma(persist_directory=directorio_db, embedding_function=embeddings)
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

with open(str(archivo_input), 'r', encoding='utf-8') as f:
    golden_dataset = json.load(f)

results = []
print(f"Iniciando inferencia RAG de {len(golden_dataset)} preguntas (k=5)...")

for i, entrada in enumerate(golden_dataset):
    pregunta = entrada['question']
    print(f"[{i+1}/{len(golden_dataset)}] Procesando...", flush=True)

    docs = retriever.invoke(pregunta)
    contextos_recuperados = [doc.page_content for doc in docs]
    respuesta = rag_chain.invoke(pregunta)

    results.append({
        "question":     pregunta,
        "answer":       respuesta,
        "contexts":     contextos_recuperados,
        "ground_truth": entrada['ground_truth']
    })

with open(str(archivo_output), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nDataset generado en '{archivo_output}' ({len(results)} entradas)")
