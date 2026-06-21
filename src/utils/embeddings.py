"""
Embedding wrapper directo sobre transformers.
Evita sentence_transformers (crash exit-5 en Windows con bge-m3 v5.x).
Produce embeddings CLS-pooled L2-normalizados — mismo comportamiento que bge-m3 por defecto.
"""
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings


class BgeM3Embeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        vectors = output.last_hidden_state[:, 0, :]
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            results.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
