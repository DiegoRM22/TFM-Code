from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vector_db = Chroma(
    persist_directory=str(DATA_DIR / "vectorstore" / "db_rgpd"),
    embedding_function=embeddings,
)

query = "¿Puedo pedir que borren mis datos personales de internet?"
resultados = vector_db.similarity_search(query, k=3)

print(f"\n--- Resultados encontrados para: '{query}' ---\n")
for i, doc in enumerate(resultados):
    print(f"FRAGMENTO {i+1} (Página {doc.metadata.get('page', '?')})")
    print(doc.page_content)
    print("-" * 50)
