import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuración de rutas
archivo_pdf = "RGPD.pdf" # CAMBIA ESTO si tu archivo se llama distinto
directorio_db = "./db_rgpd"

if not os.path.exists(archivo_pdf):
    print(f"Error: No encuentro el archivo {archivo_pdf} en esta carpeta.")
else:
    print("Cargando PDF...")
    loader = PyPDFLoader(archivo_pdf)
    paginas = loader.load()

    # 2. Segmentación (Chunking)
    # Separamos por párrafos para no romper frases legales a la mitad
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    fragmentos = text_splitter.split_documents(paginas)
    print(f"PDF dividido en {len(fragmentos)} trozos.")

    # 3. Creación de Embeddings (Modelo BGE-M3)
    print("Cargando modelo de embeddings (esto puede tardar la primera vez)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Guardar en Base de Datos Vectorial
    print("Indexando en ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=directorio_db
    )
    
    print(f"¡Hecho! Base de datos guardada en {directorio_db}")