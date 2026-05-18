import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

input_file = "ragas_dataset_limpio100.json"
output_file = "resultados_ragas_limpio100.csv"

# Importamos las métricas
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Cargar los datos
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# 2. Configurar Llama-3 como el "Juez"
evaluador_llm = ChatOllama(model="llama3:8b", temperature=0)
evaluador_wrapper = LangchainLLMWrapper(evaluador_llm)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

# 3. CONFIGURACIÓN ANTI-SATURACIÓN DE GPU
configuracion_lenta = RunConfig(max_workers=1, timeout=600)

# 4. Inicializar métricas PASANDO EL LLM a cada una
# Esto es lo que soluciona el TypeError que te ha dado
metricas = [
    Faithfulness(llm=evaluador_wrapper),
    AnswerRelevancy(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
    ContextPrecision(llm=evaluador_wrapper),
    ContextRecall(llm=evaluador_wrapper)
]

# 5. Ejecutar la evaluación
print("Iniciando evaluación secuencial con RAGAS (inyectando LLM en métricas)...")

result = evaluate(
    dataset=dataset,
    metrics=metricas,
    run_config=configuracion_lenta 
)

# 6. Mostrar y guardar resultados
print("\n--- RESULTADOS FINALES ---")
print(result)

df_resultados = result.to_pandas()
df_resultados.to_csv(output_file, index=False)
print(f"\n¡Todo listo! Resultados guardados en '{output_file}'")