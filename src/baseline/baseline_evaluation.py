import json
import sys
import pandas as pd
import warnings
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AnswerRelevancy, AnswerCorrectness

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings

DATA_DIR = PROJECT_ROOT / "data"

input_file  = DATA_DIR / "baseline" / "baseline_inference_results.json"
output_file = DATA_DIR / "baseline" / "baseline_evaluation_results.csv"

print("1. Cargando datos de inferencia baseline...")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "contexts" not in item or not item["contexts"]:
        raise ValueError(
            f"Item sin 'contexts': '{item.get('question','?')[:60]}'\n"
            "Regenera baseline_inference_results.json incluyendo los contextos."
        )
    if isinstance(item["contexts"], str):
        item["contexts"] = [item["contexts"]]

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

print("2. Configurando Juez Llama-3 (Ollama)...")
evaluador_llm = ChatOllama(model="llama3:8b", temperature=0)
evaluador_wrapper = LangchainLLMWrapper(evaluador_llm)

print("3. Cargando Embeddings (BAAI/bge-m3)...")
embeddings_wrapper = LangchainEmbeddingsWrapper(BgeM3Embeddings("BAAI/bge-m3"))

configuracion_lenta = RunConfig(max_workers=1, timeout=600)

metricas = [
    AnswerRelevancy(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
    AnswerCorrectness(llm=evaluador_wrapper, embeddings=embeddings_wrapper),
]

print("4. Iniciando evaluación RAGAS baseline...")
result = evaluate(dataset=dataset, metrics=metricas, run_config=configuracion_lenta)

print("\n" + "=" * 40)
print("RESULTADOS FINALES BASELINE")
print("=" * 40)
print(result)

df_resultados = result.to_pandas()
df_resultados.to_csv(output_file, index=False)
print(f"\nResultados guardados en: {output_file}")
