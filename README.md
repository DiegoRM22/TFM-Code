# TFM — Estudio comparativo de arquitecturas de IA Generativa para el RGPD

Comparativa experimental entre seis arquitecturas de IA Generativa (RAG, Fine-Tuning e híbridas) aplicadas a la consulta del Reglamento General de Protección de Datos (RGPD).

**[Demo con dashboard interactivo](https://tfm-code-4wdtz2ywon6nafvre7uqjt.streamlit.app)**

---

## Arquitecturas evaluadas

| Escenario | Descripción |
|---|---|
| S1 (Baseline)| Llama-3-8B-Instruct sin modificar |
| S2a (RAG Base) | ChromaDB + BGE-M3 + Llama-3 |
| S2b (RAG Avanzado) | BM25+vectorial + reranking BGE + HyDE |
| S3 (Fine-Tuning)| QLoRA 4-bit sobre Llama-3-8B-Instruct |
| S4 (Híbrido Base)| Retriever S2a + generador S3 |
| S4b (Híbrido Avanzado) | Retriever S2b + generador S3 |

Evaluación con RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness) sobre un Golden Dataset de 92 pares pregunta-respuesta sobre el RGPD, con GPT-4o-mini como juez evaluador.

---

## Estructura del proyecto

```
TFM/
├── src/
│   ├── baseline/          # Inferencia y evaluación S1
│   ├── fine_tuning/       # Entrenamiento QLoRA, inferencia y evaluación S3
│   ├── hybrid/            # Arquitecturas híbridas S4 y S4b
│   ├── rag/               # Pipelines RAG S2a y S2b
│   ├── preprocessing/     # Extracción PDF, chunking e indexación
│   └── utils/             # Dashboard Streamlit y utilidades
├── data/
│   ├── shared/            # Golden Dataset y test set
│   ├── vectorstore/       # ChromaDB (pymupdf y semántico)
│   ├── rag/               # Resultados RAG
│   ├── fine_tuning/       # Resultados Fine-Tuning
│   ├── hybrid/            # Resultados híbridos
│   └── baseline/          # Resultados baseline
├── models/                # Checkpoints y adaptadores LoRA
└── notebooks/             # Notebooks de Colab para ejecución en GPU
```

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| LLM base | meta-llama/Meta-Llama-3-8B-Instruct |
| LLM local | llama3:8b vía Ollama |
| Embeddings | BAAI/bge-m3 |
| Vector DB | ChromaDB |
| Reranking | BAAI/bge-reranker-v2-m3 |
| Fine-tuning | QLoRA 4-bit, PEFT + TRL + BitsAndBytes |
| Evaluación | RAGAS + GPT-4o-mini como juez |
| Dashboard | Streamlit + Plotly |

---

## Instalación

```bash
# Modelo local
ollama pull llama3:8b

# Entorno Python
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Dependencias
pip install langchain langchain-community langchain-ollama
pip install chromadb ragas pandas streamlit plotly
pip install pymupdf ftfy python-dotenv langchain-openai
```

---

## Ejecución

```bash
# 1. Indexar corpus
python src/preprocessing/preprocessing_clean.py

# 2. Generar Golden Dataset
python src/preprocessing/golden_dataset_generator.py

# 3. Inferencia (en este caso para el RAG base)
python src/rag/rag_inference.py

# 4. Evaluación con GPT
python src/rag/rag_evaluation_test_gpt.py

# Dashboard
streamlit run src/utils/dashboard_tfm_v2.py
```

El fine-tuning requiere GPU con mínimo 8GB VRAM. Se recomienda Google Colab (T4/A100); los notebooks están disponibles en `notebooks/`.
