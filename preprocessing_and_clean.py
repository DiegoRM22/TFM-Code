import fitz  # PyMuPDF
import ftfy  # Fixes text for you
import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def extraer_y_limpiar_pdf(ruta_pdf):
    print(f"Extrayendo texto con PyMuPDF desde: {ruta_pdf}")
    doc_fitz = fitz.open(ruta_pdf)
    documentos_langchain = []

    for i, pagina in enumerate(doc_fitz):
        # Extraemos el texto de la página
        texto = pagina.get_text("text") # Opción "text" mantiene el orden lógico
        
        # --- PROCESO DE LIMPIEZA ---
        # 1. Reparar codificación y caracteres extraños
        texto = ftfy.fix_text(texto)
        
        # 2. Unir palabras cortadas por guion al final de línea (ej: regla- mento)
        texto = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', texto)
        
        # 3. Eliminar saltos de línea innecesarios y normalizar espacios
        texto = texto.replace('\n', ' ')
        texto = re.sub(r'\s+', ' ', texto)
        
        # 4. (Opcional) Unir letras sueltas persistentes (f alsear -> falsear)
        # Solo lo aplicamos si detectamos que el PDF sigue dando problemas
        texto = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', texto) 

        # Crear objeto Document para LangChain
        nuevo_doc = Document(
            page_content=texto.strip(),
            metadata={"source": ruta_pdf, "page": i + 1}
        )
        documentos_langchain.append(nuevo_doc)
    
    return documentos_langchain

# --- FLUJO PRINCIPAL ---
archivo_pdf = "RGPD.pdf"
directorio_db = "./db_rgpd_pymupdf"

if not os.path.exists(archivo_pdf):
    print("Error: No se encuentra el archivo.")
else:
    # 1. Carga y Limpieza
    documentos_limpios = extraer_y_limpiar_pdf(archivo_pdf)

    # 2. Chunking (ahora con el texto mucho más fluido)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[". ", "\n", " ", ""]
    )
    fragmentos = text_splitter.split_documents(documentos_limpios)

    # 3. Embeddings e Indexación
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=directorio_db
    )
    print(f"¡BDD finalizada con PyMuPDF en {directorio_db}!")