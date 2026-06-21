import fitz
import ftfy
import re
import os
import sys
from pathlib import Path
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings

DATA_DIR = PROJECT_ROOT / "data"

archivo_pdf   = str(PROJECT_ROOT / "RGPD.pdf")
directorio_db = str(DATA_DIR / "vectorstore" / "db_rgpd_semantic")


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
    print("Error: No se encuentra el archivo RGPD.pdf")
else:
    documentos_limpios = extraer_y_limpiar_pdf(archivo_pdf)

    print("Cargando BgeM3Embeddings para SemanticChunker...")
    embeddings = BgeM3Embeddings("BAAI/bge-m3")

    # SemanticChunker agrupa frases por similitud semántica en lugar de tamaño fijo
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )

    print("Generando chunks semánticos (puede tardar varios minutos)...")
    fragmentos = text_splitter.split_documents(documentos_limpios)
    print(f"Total fragmentos semánticos: {len(fragmentos)}")

    vector_db = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=directorio_db,
    )
    print(f"BDD finalizada con chunking semántico en {directorio_db}!")
