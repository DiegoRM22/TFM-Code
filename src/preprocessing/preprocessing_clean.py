import fitz
import ftfy
import re
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

archivo_pdf   = str(PROJECT_ROOT / "RGPD.pdf")
directorio_db = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")


def extraer_y_limpiar_pdf(ruta_pdf):
    print(f"Extrayendo texto con PyMuPDF desde: {ruta_pdf}")
    doc_fitz = fitz.open(ruta_pdf)
    documentos_langchain = []

    for i, pagina in enumerate(doc_fitz):
        texto = pagina.get_text("text")
        texto = ftfy.fix_text(texto)
        texto = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', texto)
        texto = texto.replace('\n', ' ')
        texto = re.sub(r'\s+', ' ', texto)
        texto = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', texto)

        documentos_langchain.append(Document(
            page_content=texto.strip(),
            metadata={"source": ruta_pdf, "page": i + 1}
        ))

    return documentos_langchain


if not os.path.exists(archivo_pdf):
    print("Error: No se encuentra el archivo.")
else:
    documentos_limpios = extraer_y_limpiar_pdf(archivo_pdf)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[". ", "\n", " ", ""]
    )
    fragmentos = text_splitter.split_documents(documentos_limpios)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=directorio_db,
    )
    print(f"¡BDD finalizada con PyMuPDF en {directorio_db}!")
