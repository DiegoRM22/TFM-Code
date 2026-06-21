"""
Evaluación RAGAS híbrido avanzado GPT v3 (sin HyDE).
Input:  data/hybrid/advanced/hybrid_advanced_inference_results_v3.json
Output: data/hybrid/advanced/hybrid_advanced_evaluation_gpt_results_v3.csv
"""
import json
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from unittest.mock import MagicMock
for _mod in ["langchain_community.chat_models.vertexai", "langchain_community.llms.vertexai"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY no definida.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "hybrid" / "advanced" / "hybrid_advanced_inference_results_v3.json"
output_file = DATA_DIR / "hybrid" / "advanced" / "hybrid_advanced_evaluation_gpt_results_v3.csv"

print("1. Cargando resultados de inferencia hibrida avanzada (v3)...")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "contexts" not in item or not item["contexts"]:
        raise ValueError(f"Item sin 'contexts': '{item.get('question','?')[:60]}'")
    if isinstance(item["contexts"], str):
        item["contexts"] = [item["contexts"]]

for item in data:
    if "question"     in item: item["user_input"]          = item.pop("question")
    if "answer"       in item: item["response"]             = item.pop("answer")
    if "contexts"     in item: item["retrieved_contexts"]   = item.pop("contexts")
    if "ground_truth" in item: item["reference"]            = item.pop("ground_truth")

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

print("2. Configurando juez GPT-4o-mini (OpenAI)...")
evaluador_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluador_wrapper = LangchainLLMWrapper(evaluador_llm)

print("3. Cargando embeddings text-embedding-3-small (OpenAI)...")
embeddings_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

configuracion = RunConfig(max_workers=4, timeout=120)

metricas = [
    Faithfulness(llm=evaluador_wrapper),
    AnswerRelevancy(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
    ContextPrecision(llm=evaluador_wrapper),
    ContextRecall(llm=evaluador_wrapper),
    AnswerCorrectness(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
]

print(f"4. Iniciando evaluacion RAGAS hibrido avanzado GPT v3 ({len(data)}q x 5 metricas)...")
result = evaluate(dataset=dataset, metrics=metricas, run_config=configuracion)

print("\n" + "=" * 40)
print("RESULTADOS FINALES HIBRIDO AVANZADO v3 (GPT-4o-mini)")
print("=" * 40)
print(result)

df_resultados = result.to_pandas()
df_resultados.to_csv(output_file, index=False)
print(f"\nResultados guardados en: {output_file}")
