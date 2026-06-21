from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

db_path = DATA_DIR / "vectorstore" / "db_rgpd"
if not db_path.exists():
    print("Error: No se encuentra la carpeta 'data/vectorstore/db_rgpd'. Ejecuta primero el preprocesamiento.")
    exit()

vector_db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)

llm = OllamaLLM(model="llama3:8b")

template = """Eres un consultor jurídico especializado en el RGPD. Responde de forma DIRECTA y CONCISA.
REGLAS:
- Usa ÚNICAMENTE la información del contexto proporcionado.
- Si el artículo exacto aparece en el contexto, cítalo.
- Si la respuesta no está en el contexto, responde: "El contexto no contiene información suficiente para responder esta pregunta."
- No añadas información que no esté en el contexto.
- Responde siempre en español.

Contexto legal del RGPD:
{context}

Pregunta del usuario: {question}

Respuesta directa (máximo 3-4 frases):"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | QA_PROMPT
    | llm
    | StrOutputParser()
)

print("\n--- CONSULTOR RGPD LISTO ---")
pregunta = "¿En qué casos no es necesario el consentimiento para tratar datos?"

print(f"\nConsultando: {pregunta}")
try:
    docs = retriever.invoke(pregunta)
    respuesta = qa_chain.invoke(pregunta)
    print("\nRESPUESTA DE LLAMA 3:")
    print(respuesta)
    print("\nFUENTES UTILIZADAS:")
    for doc in docs:
        print(f"- {doc.metadata.get('source', 'RGPD')} (Página {doc.metadata.get('page', '?')})")
except Exception as e:
    print(f"Error al generar la respuesta: {e}")
