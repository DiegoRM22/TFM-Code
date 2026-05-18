import os
import json
import pandas as pd
import warnings
import torch

# 1. Configuración de Dataset e Imports
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import AnswerRelevancy, AnswerCorrectness

warnings.filterwarnings("ignore")

input_file = "resultados_para_ragas.json"
output_file = "metricas_finetuning_resultados.csv"

# 2. Cargar datos
print("1. Cargando datos de inferencia...")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    item["contexts"] = [item["ground_truth"]]

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# 3. Configurar Evaluadores
print("2. Configurando Juez Llama-3 (Ollama)...")
evaluador_llm = ChatOllama(model="llama3:8b", temperature=0)
evaluador_wrapper = LangchainLLMWrapper(evaluador_llm)

print("3. Cargando Embeddings (all-MiniLM-L6-v2) - Formato seguro Safetensors...")
# Cambiamos a este modelo porque garantiza compatibilidad total sin errores de seguridad
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

# 4. Configuración RAGAS
configuracion_lenta = RunConfig(max_workers=1, timeout=600)

metricas = [
    AnswerRelevancy(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
    AnswerCorrectness(llm=evaluador_wrapper, embeddings=embeddings_wrapper)
]

# 5. Ejecución
print("4. Iniciando evaluación RAGAS (esto puede tardar unos minutos)...")
result = evaluate(
    dataset=dataset,
    metrics=metricas,
    run_config=configuracion_lenta 
)

print("\n" + "="*40)
print("RESULTADOS FINALES FINE-TUNING")
print("="*40)
print(result)

# Guardar resultados
df_resultados = result.to_pandas()
df_resultados.to_csv(output_file, index=False)
print(f"\n¡Proceso completado! Resultados guardados en: {output_file}")