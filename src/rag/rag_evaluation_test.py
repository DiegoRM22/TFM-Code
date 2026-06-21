"""
Evaluación RAGAS del RAG sobre test set (14q) — comparación directa con FT, baseline e híbrido.
"""
import json
import sys
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings

DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "rag" / "rag_inference_test_results.json"
output_file = DATA_DIR / "rag" / "rag_evaluation_test_results.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

evaluador_llm = ChatOllama(model="llama3:8b", temperature=0)
evaluador_wrapper = LangchainLLMWrapper(evaluador_llm)
embeddings_wrapper = LangchainEmbeddingsWrapper(BgeM3Embeddings("BAAI/bge-m3"))

configuracion_lenta = RunConfig(max_workers=1, timeout=600)

metricas = [
    Faithfulness(llm=evaluador_wrapper),
    AnswerRelevancy(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
    ContextPrecision(llm=evaluador_wrapper),
    ContextRecall(llm=evaluador_wrapper),
]

print("Iniciando evaluación RAGAS — RAG test set (14q × 4 métricas)...")
result = evaluate(dataset=dataset, metrics=metricas, run_config=configuracion_lenta)

print("\n--- RESULTADOS RAG TEST SET ---")
print(result)

df_resultados = result.to_pandas()
df_resultados.to_csv(output_file, index=False)
print(f"\nResultados guardados en '{output_file}'")
