import os
import json
import re
from langchain_core.prompts import PromptTemplate

# IMPORTS ACTUALIZADOS (Sin Deprecation Warnings)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. Configuración
directorio_db = "./db_rgpd_pymupdf"
archivo_salida = "golden_dataset_raw_limpio100.json"

# Cargamos el modelo LLM local
print("Iniciando Llama-3...")
llm = OllamaLLM(model="llama3:8b", temperature=0.0) 

print("Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}  # <--- ESTO DEBE SER 'cpu'
)
print("Conectando a ChromaDB...")
vector_db = Chroma(persist_directory=directorio_db, embedding_function=embeddings)

# 2. Extraer fragmentos de la base de datos
documentos = vector_db.get()
textos_rgpd = documentos['documents']
print(f"Se han recuperado {len(textos_rgpd)} fragmentos del RGPD.")

# Función para limpiar el texto del PDF (quita saltos de línea rotos)
def limpiar_texto(texto):
    texto = texto.replace('\n', ' ')
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# 3. Definir el Prompt para el LLM (Mejorado para dar variedad)
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

# 4. Bucle de Generación
dataset = []
limite_ejemplos = 100 # Empezamos con 10 para probar que todo va perfecto

print(f"\nGenerando {limite_ejemplos} pares de Pregunta/Respuesta...")

for i, texto_bruto in enumerate(textos_rgpd[:limite_ejemplos]):
    print(f"Procesando fragmento {i+1}/{limite_ejemplos}...")
    
    texto_limpio = limpiar_texto(texto_bruto)
    
    try:
        prompt_formateado = prompt_template.format(contexto=texto_limpio)
        respuesta_llm = llm.invoke(prompt_formateado)
        
        # Extracción "Acorazada" usando Expresiones Regulares
        # Busca cualquier cosa entre { y }
        match = re.search(r'\{.*?\}', respuesta_llm, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            par_qa = json.loads(json_str)
            par_qa["contexts"] = [texto_limpio] # Guardamos el contexto limpio
            dataset.append(par_qa)
            print(f"  ✓ Generado con éxito: {par_qa['question'][:50]}...")
        else:
            print(f"  -> Error: No se encontró estructura JSON en el fragmento {i+1}")
            print(f"  -> Respuesta del modelo fue: {respuesta_llm}")

    except json.JSONDecodeError as e:
        print(f"  -> Error de formato JSON en fragmento {i+1}: {e}")
    except Exception as e:
        print(f"  -> Error inesperado en fragmento {i+1}: {e}")

# 5. Guardar el resultado
with open(archivo_salida, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"\n¡Proceso terminado! Se ha generado el archivo '{archivo_salida}' con {len(dataset)} registros validos.")