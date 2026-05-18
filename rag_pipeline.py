import os
# Importaciones actualizadas para LangChain 0.2/0.3
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Configuración de Componentes
print("Iniciando sistema RAG legal...")

# Modelo de embeddings (el mismo que usaste en el preprocesamiento)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Conectamos a la base de datos que ya creaste
if not os.path.exists("./db_rgpd"):
    print("Error: No se encuentra la carpeta 'db_rgpd'. Ejecuta primero el preprocesamiento.")
    exit()

vector_db = Chroma(persist_directory="./db_rgpd", embedding_function=embeddings)

# Conectamos a Llama 3 vía Ollama
# Asegúrate de tener Ollama abierto en tu PC
llm = OllamaLLM(model="llama3:8b")

# 2. Definición del Prompt (Estructura de respuesta legal)
template = """Eres un consultor jurídico experto en el RGPD. 
Tu objetivo es responder de forma clara y rigurosa basándote SOLO en el contexto proporcionado.

Contexto legal extraído del RGPD:
{context}

Pregunta del usuario: {question}

Respuesta (cita artículos si es posible):"""

QA_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# 3. Creación de la cadena de respuesta
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# 4. Interfaz de consulta
print("\n--- CONSULTOR RGPD LISTO ---")
pregunta = "¿En qué casos no es necesario el consentimiento para tratar datos?"

print(f"\nConsultando: {pregunta}")
try:
    respuesta = qa_chain.invoke({"query": pregunta})
    print("\n🤖 RESPUESTA DE LLAMA 3:")
    print(respuesta["result"])
    
    print("\n📚 FUENTES UTILIZADAS:")
    for doc in respuesta["source_documents"]:
        print(f"- {doc.metadata.get('source', 'RGPD')} (Página {doc.metadata.get('page', '?')})")
except Exception as e:
    print(f"Error al generar la respuesta: {e}")