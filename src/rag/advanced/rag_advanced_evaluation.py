"""
Evaluación RAGAS del RAG avanzado S2b (EnsembleRetriever + reranking + HyDE).
"""
import json
import sys
import pandas as pd
from pathlib import Path
from datasets import Dataset
from unittest.mock import MagicMock
for _mod in ["langchain_community.chat_models.vertexai", "langchain_community.llms.vertexai"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings

DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "rag" / "advanced" / "rag_advanced_inference_dataset.json"
output_file = DATA_DIR / "rag" / "advanced" / "rag_advanced_evaluation_results.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "question"     in item: item["user_input"]          = item.pop("question")
    if "answer"       in item: item["response"]             = item.pop("answer")
    if "contexts"     in item: item["retrieved_contexts"]   = item.pop("contexts")
    if "ground_truth" in item: item["reference"]            = item.pop("ground_truth")

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
    AnswerCorrectness(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
]

print("Iniciando evaluación RAGAS — RAG avanzado S2b (14q × 5 métricas)...")
result = evaluate(dataset=dataset, metrics=metricas, run_config=configuracion_lenta)

print("\n--- RESULTADOS RAG AVANZADO S2b ---")
print(result)

df_resultados = result.to_pandas()
df_resultados.to_csv(output_file, index=False)
print(f"\nResultados guardados en '{output_file}'")
