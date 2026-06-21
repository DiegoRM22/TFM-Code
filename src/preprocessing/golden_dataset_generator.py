import os
import json
import re
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

directorio_db = str(DATA_DIR / "vectorstore" / "db_rgpd_pymupdf")
archivo_salida = str(DATA_DIR / "shared" / "golden_dataset_raw_limpio100.json")

print("Iniciando Llama-3...")
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

print("Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}
)
print("Conectando a ChromaDB...")
vector_db = Chroma(persist_directory=directorio_db, embedding_function=embeddings)

documentos = vector_db.get()
textos_rgpd = documentos['documents']
print(f"Se han recuperado {len(textos_rgpd)} fragmentos del RGPD.")


def limpiar_texto(texto):
    texto = texto.replace('\n', ' ')
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


prompt_template = PromptTemplate(
    input_variables=["contexto"],
    template="""Eres un experto jurídico en el Reglamento General de Protección de Datos (RGPD).
Lee el siguiente fragmento legal y diseña UNA pregunta de examen y su respuesta perfecta.

Fragmento:
"{contexto}"

REGLAS ESTRICTAS:
1. Varía el tipo de pregunta. No uses siempre "¿Cuál es el objetivo?". Usa formulaciones como: "¿Qué obligaciones tiene...?", "¿Qué se entiende por...?", "¿En qué casos aplica...?", "¿Quién es el responsable de...?".
2. La respuesta (ground_truth) debe ser exacta, profesional y basarse ÚNICAMENTE en el fragmento.
3. TU RESPUESTA DEBE SER ÚNICA Y EXCLUSIVAMENTE UN OBJETO JSON VÁLIDO. No añadas introducciones, ni notas, ni formato markdown.

Devuelve SOLO este formato exacto:
{{
  "question": "tu pregunta variada aquí",
  "ground_truth": "tu respuesta detallada aquí"
}}
"""
)

dataset = []
limite_ejemplos = 100

print(f"\nGenerando {limite_ejemplos} pares de Pregunta/Respuesta...")

for i, texto_bruto in enumerate(textos_rgpd[:limite_ejemplos]):
    print(f"Procesando fragmento {i+1}/{limite_ejemplos}...")
    texto_limpio = limpiar_texto(texto_bruto)

    try:
        respuesta_llm = llm.invoke(prompt_template.format(contexto=texto_limpio))
        match = re.search(r'\{.*?\}', respuesta_llm, re.DOTALL)

        if match:
            par_qa = json.loads(match.group(0))
            par_qa["contexts"] = [texto_limpio]
            dataset.append(par_qa)
            print(f"  OK: {par_qa['question'][:50]}...")
        else:
            print(f"  Error: No se encontró JSON en fragmento {i+1}")

    except json.JSONDecodeError as e:
        print(f"  Error JSON en fragmento {i+1}: {e}")
    except Exception as e:
        print(f"  Error en fragmento {i+1}: {e}")

with open(archivo_salida, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"\n¡Terminado! '{archivo_salida}' con {len(dataset)} registros válidos.")
