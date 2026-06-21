# Resultados TFM — RAG + Fine-tuning para RGPD

## Experimentos

Se evalúan 4 arquitecturas sobre documentación del RGPD usando métricas RAGAS.
La comparación principal se hace sobre el **test set de 14 preguntas** compartido (mismo split, comparación justa).

---

## Métricas RAGAS — comparación directa (14 preguntas test set)

| Arquitectura | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|---|
| RAG puro | 0.9524 | 0.7913 | 0.9765 | 0.9107 | — |
| Híbrido RAG+FT | 0.8462 | 0.7741 | 0.9778 | **1.0000** | — |
| Fine-tuning (LoRA) | — | 0.7224 | — | — | 0.7689 |
| Baseline LLaMA-3 | — | 0.6584 | — | — | 0.5908 |

**n = 14 para todas las arquitecturas** (mismo test set).

---

## Observaciones

### RAG puro vs Híbrido
- RAG puro supera en Faithfulness (+0.106) y Answer Relevancy (+0.017)
- Híbrido logra Context Recall perfecto (1.0000) — recupera todos los fragmentos relevantes
- Context Precision prácticamente idéntica (0.9765 vs 0.9778)
- El generador fine-tuneado no mejora la generación respecto al RAG puro con Ollama en este dominio

### Fine-tuning vs Baseline
- Fine-tuning mejora Baseline en **+0.064 Answer Relevancy** (+9.7%)
- Fine-tuning mejora Baseline en **+0.178 Answer Correctness** (+30.1%)
- El LoRA domain-specific aporta mejora significativa en corrección factual

### Ranking por Answer Relevancy (métrica común)
1. RAG puro: **0.7913**
2. Híbrido RAG+FT: **0.7741**
3. Fine-tuning: 0.7224
4. Baseline: 0.6584

---

## RAG Pipeline — caracterización completa (92 preguntas)

Evaluación del pipeline RAG sobre el conjunto completo de 92 preguntas generado automáticamente.
No comparable directamente con las arquitecturas anteriores (conjunto diferente).

| Métrica | Media |
|---|---|
| Faithfulness | 0.8872 |
| Answer Relevancy | 0.7647 |
| Context Precision | 0.9738 |
| Context Recall | 0.9149 |

**n = 92** — mismo pipeline RAG, conjunto más amplio y variado.

---

## Configuración experimental

### Datos
- **golden_dataset_raw_limpio100.json**: 92 preguntas generadas con GPT-4 sobre el RGPD
- **dataset_test.json**: 14 preguntas curadas, formato JSONL Llama-3 (split de test)

### Embeddings y retrieval
- Modelo: `BAAI/bge-m3` (custom `BgeM3Embeddings` via transformers)
- Base de datos vectorial: ChromaDB (`db_rgpd_pymupdf`, parser PyMuPDF)
- k = 5 documentos recuperados

### Generadores
| Arquitectura | Generador |
|---|---|
| RAG puro | `llama3:8b` via Ollama, temperatura 0 |
| Fine-tuning | Meta-Llama-3-8B-Instruct + LoRA (QLoRA 4-bit NF4, r=16, α=32, checkpoint-120) |
| Baseline | Meta-Llama-3-8B-Instruct sin LoRA |
| Híbrido | ChromaDB retrieval + Fine-tuning generador |

### Evaluador RAGAS
- LLM juez: Llama-3-8B-Instruct 4-bit (Colab T4) / `llama3:8b` Ollama (local)
- Embeddings: BAAI/bge-m3
- RAGAS v0.2.12, `RunConfig(max_workers=1-2, timeout=300-600)`

---

## Estructura de resultados

```
data/
├── rag/
│   ├── rag_evaluation_results.csv          # RAG 92q ✓
│   └── rag_evaluation_test_results.csv     # RAG 14q ✓
├── fine_tuning/
│   └── fine_tuning_evaluation_results.csv  # FT 14q ✓
├── baseline/
│   └── baseline_evaluation_results.csv     # Baseline 14q ✓
└── hybrid/
    └── hybrid_evaluation_results.csv       # Híbrido 14q ✓
```
