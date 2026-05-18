import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Configuración
archivo_input = "golden_dataset_raw_limpio100.json"
archivo_output = "ragas_dataset_limpio100.json"
directorio_db = "./db_rgpd_pymupdf"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}
)

# LLM con parámetros para mayor consistencia
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

vector_db = Chroma(persist_directory=directorio_db, embedding_function=embeddings)

# 2. DEFINIR EL PROMPT EN ESPAÑOL (Esto es lo que faltaba)
template = """Eres un asistente legal experto en el RGPD. 
Responde a la siguiente pregunta utilizando ÚNICAMENTE el contexto proporcionado. 
Si la respuesta no está en el contexto, di que no lo sabes, no inventes nada.
IMPORTANTE: Tu respuesta debe ser TOTALMENTE EN ESPAÑOL.

Contexto: {context}

Pregunta: {question}

Respuesta en español:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# 3. Configurar la cadena con el nuevo Prompt
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT} # Aplicamos el prompt aquí
)

# 4. Cargar y Procesar
with open(archivo_input, 'r', encoding='utf-8') as f:
    golden_dataset = json.load(f)

results = []
print(f"Iniciando regeneración en español de {len(golden_dataset)} registros...")

for i, entrada in enumerate(golden_dataset):
    pregunta = entrada['question']
    print(f"[{i+1}/100] Procesando en español...")
    
    respuesta_rag = rag_chain.invoke({"query": pregunta})
    
    contextos_recuperados = [doc.page_content for doc in respuesta_rag['source_documents']]
    
    results.append({
        "question": pregunta,
        "answer": respuesta_rag['result'],
        "contexts": contextos_recuperados,
        "ground_truth": entrada['ground_truth']
    })

# 5. Guardar
with open(archivo_output, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Dataset v2 generado con éxito en '{archivo_output}'")