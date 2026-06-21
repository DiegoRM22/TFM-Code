"""
Retriever avanzado autocontenido: RRF(BM25 + Chroma) + CrossEncoder reranking + HyDE.
Evita imports de langchain_classic.retrievers (sentence_transformers crash en Windows).
"""
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.embeddings import BgeM3Embeddings


class AdvancedRetriever:
    """
    EnsembleRetriever (BM25 0.4 + Chroma 0.6) con RRF manual
    + CrossEncoder BAAI/bge-reranker-v2-m3 (top-10 → top-5).
    """

    def __init__(self, db_dir: str, top_n: int = 5, k_rrf: int = 60):
        self.top_n = top_n
        self.k_rrf = k_rrf

        print("  Cargando embeddings bge-m3...")
        self.embeddings = BgeM3Embeddings("BAAI/bge-m3")
        self.vector_db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)

        print("  Extrayendo documentos para BM25...")
        raw = self.vector_db.get()
        bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        self.bm25 = BM25Retriever.from_documents(bm25_docs, k=10)

        print("  Cargando CrossEncoder BAAI/bge-reranker-v2-m3...")
        self._tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        self._reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        self._reranker.eval()

    def _rrf_fuse(self, bm25_docs: List[Document], chroma_docs: List[Document]) -> List[Document]:
        """Reciprocal Rank Fusion con pesos 0.4 / 0.6."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 0.4 / (self.k_rrf + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(chroma_docs):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 0.6 / (self.k_rrf + rank + 1)
            doc_map[key] = doc

        ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [doc_map[k] for k in ranked_keys[:10]]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """CrossEncoder reranking — devuelve top_n documentos."""
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        with torch.no_grad():
            inputs = self._tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            logits = self._reranker(**inputs).logits.squeeze(-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            scores = logits.tolist()

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_n]]

    def invoke(self, query: str) -> List[Document]:
        bm25_results = self.bm25.invoke(query)
        chroma_results = self.vector_db.similarity_search(query, k=10)
        fused = self._rrf_fuse(bm25_results, chroma_results)
        return self._rerank(query, fused)


def build_advanced_retriever(db_dir: str) -> AdvancedRetriever:
    return AdvancedRetriever(db_dir)


def hyde_expand(question: str, llm) -> str:
    """HyDE: genera documento hipotético para mejorar recuperación densa."""
    prompt = (
        "Genera un párrafo breve de documentación legal del RGPD que respondería "
        "directamente a esta pregunta. Escribe solo el fragmento, sin preámbulo ni título.\n\n"
        f"Pregunta: {question}\nFragmento hipotético:"
    )
    return llm.invoke(prompt).strip()
