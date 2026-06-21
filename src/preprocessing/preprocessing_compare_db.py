from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

path_sucia = str(DATA_DIR / "vectorstore" / "db_rgpd")
path_limpia = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")

print("Conectando con las bases de datos...")
db_sucia  = Chroma(persist_directory=path_sucia,  embedding_function=embeddings)
db_limpia = Chroma(persist_directory=path_limpia, embedding_function=embeddings)


def comparar_fragmentos(query_text):
    print(f"\n{'='*60}")
    print(f" BUSCANDO: '{query_text}'")
    print("=" * 60)

    docs_sucios  = db_sucia.similarity_search(query_text,  k=1)
    docs_limpios = db_limpia.similarity_search(query_text, k=1)

    print("\n>>> BASE DE DATOS ORIGINAL (SUCIA):")
    print("-" * 30)
    if docs_sucios:
        print(docs_sucios[0].page_content[:600])

    print("\n\n>>> BASE DE DATOS PROCESADA (PyMuPDF):")
    print("-" * 30)
    if docs_limpios:
        print(docs_limpios[0].page_content[:600])


comparar_fragmentos("obstáculo al ejercicio de las actividades económicas impedir")
