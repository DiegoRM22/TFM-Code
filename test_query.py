from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Cargamos el mismo modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 2. Conectamos con la base de datos que ya creaste
vector_db = Chroma(persist_directory="./db_rgpd", embedding_function=embeddings)

# 3. Hacemos una pregunta técnica
query = "¿Puedo pedir que borren mis datos personales de internet?"
# k=3 nos devuelve los 3 fragmentos más parecidos
resultados = vector_db.similarity_search(query, k=3)

print(f"\n--- Resultados encontrados para: '{query}' ---\n")
for i, doc in enumerate(resultados):
    print(f"FRAGMENTO {i+1} (Página {doc.metadata.get('page', '?')})")
    print(doc.page_content)
    print("-" * 50)