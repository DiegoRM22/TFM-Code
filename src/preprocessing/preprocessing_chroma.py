import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

archivo_pdf  = str(PROJECT_ROOT / "RGPD.pdf")
directorio_db = str(DATA_DIR / "vectorstore" / "db_rgpd")

if not os.path.exists(archivo_pdf):
    print(f"Error: No encuentro el archivo {archivo_pdf}")
else:
    loader = PyPDFLoader(archivo_pdf)
    paginas = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    fragmentos = text_splitter.split_documents(paginas)
    print(f"PDF dividido en {len(fragmentos)} trozos.")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'}
    )

    vector_db = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=directorio_db,
    )
    print(f"¡Hecho! Base de datos guardada en {directorio_db}")
