# Asistente RAG Legal para el RGPD — TFM

Sistema RAG (Retrieval-Augmented Generation) con fine-tuning de Llama-3-8B para consultas en lenguaje natural sobre el Reglamento General de Protección de Datos (RGPD). Todo el procesamiento es **100% local**: sin APIs externas, sin datos en la nube.

---

## Estructura del Proyecto

```
TFM/
├── src/                          # Scripts Python
├── data/                         # Datasets, resultados y bases de datos vectoriales
├── models/                       # Checkpoints y adaptadores LoRA entrenados
├── RGPD.pdf                      # Documento fuente (no incluido en el repo)
└── venv/                         # Entorno virtual Python
```

---

## Arquitectura: Dos Pipelines

El proyecto implementa y compara dos enfoques para responder preguntas sobre el RGPD:

```
                          RGPD.pdf
                              │
               ┌──────────────┴───────────────┐
               │                              │
    preprocessing.py              preprocessing_and_clean.py
    (PyPDFLoader, sin limpiar)    (PyMuPDF + ftfy, limpio)
               │                              │
          data/db_rgpd/               data/db_rgpd_pymupdf/
                                              │
    ┌─────────────────────────────────────────┤
    │                                         │
    │   PIPELINE A: RAG Baseline              │
    │                                         │
    │   golden_dataset_generator.py ──────────┘
    │          │
    │   data/golden_dataset_raw_limpio100.json
    │          │
    │   rag_inference_ragas.py
    │   (RAG: Llama3 + ChromaDB)
    │          │
    │   data/ragas_dataset_limpio100.json
    │          │
    │   ragas_evaluation.py
    │          │
    │   data/resultados_ragas_limpio100.csv
    │
    └─── PIPELINE B: Fine-Tuning
    
    data/golden_dataset_raw_limpio100.json
               │
        fine_tuning_rgpd.py
        (QLoRA 4-bit, SFTTrainer)
          │              │
    data/dataset_test.json   models/llama3_rgpd_lora/
          │              models/resultados_rgpd_llama3/
          │
       batch_eval.py
       (inferencia batch, checkpoint-200)
          │
    data/resultados_para_ragas.json
          │
    fine_tuning_ragas_evaluation.py
          │
    data/metricas_finetuning_resultados.csv
```

---

## Scripts (`src/`)

### Preprocesamiento

| Script | Descripcion | Lee | Escribe |
|--------|-------------|-----|---------|
| `preprocessing.py` | Carga el PDF con PyPDFLoader, divide en chunks (1000 chars, overlap 150), genera embeddings BAAI/bge-m3 e indexa en ChromaDB | `RGPD.pdf` | `data/db_rgpd/` |
| `preprocessing_and_clean.py` | Alternativa mejorada: usa PyMuPDF + ftfy para extraer texto limpio (repara codificacion, une palabras cortadas por guion, normaliza espacios) antes de indexar | `RGPD.pdf` | `data/db_rgpd_pymupdf/` |
| `comparacion_bbdd.py` | Herramienta de diagnostico: lanza la misma query a ambas BBDDs y muestra los primeros 600 chars del resultado para comparar calidad de extraccion | `data/db_rgpd/`, `data/db_rgpd_pymupdf/` | — |

### Pipeline A: RAG Baseline

| Script | Descripcion | Lee | Escribe |
|--------|-------------|-----|---------|
| `golden_dataset_generator.py` | Genera el golden dataset: recupera fragmentos de ChromaDB, usa Llama-3 via Ollama para crear pares `{question, ground_truth, contexts}` en formato JSON. Limite: 100 ejemplos. Extraccion JSON robusta con regex. | `data/db_rgpd_pymupdf/` | `data/golden_dataset_raw_limpio100.json` |
| `rag_inference_ragas.py` | Ejecuta el pipeline RAG sobre cada pregunta del golden dataset: recupera 3 contextos de ChromaDB, genera respuesta con Llama-3. Produce el dataset listo para evaluacion RAGAS. | `data/golden_dataset_raw_limpio100.json`, `data/db_rgpd_pymupdf/` | `data/ragas_dataset_limpio100.json` |
| `ragas_evaluation.py` | Evalua el RAG con 4 metricas RAGAS: `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`. LLM juez: Llama-3 via Ollama. Embeddings: BAAI/bge-m3. Configuracion secuencial (max_workers=1) para evitar saturar GPU. | `data/ragas_dataset_limpio100.json` | `data/resultados_ragas_limpio100.csv` |

### Pipeline B: Fine-Tuning

| Script | Descripcion | Lee | Escribe |
|--------|-------------|-----|---------|
| `fine_tuning_rgpd.py` | Fine-tuning de Llama-3-8B-Instruct con QLoRA 4-bit (nf4, bfloat16). Split 70/15/15 (train/val/test). LoRA: r=16, alpha=32, capas q/k/v/o_proj. 200 pasos de entrenamiento, batch=1, grad_accum=4. Requiere `HF_TOKEN` en `.env`. | `data/golden_dataset_raw_limpio100.json` | `data/dataset_test.json`, `models/resultados_rgpd_llama3/` (checkpoints), `models/llama3_rgpd_lora/` (adaptadores finales) |
| `batch_eval.py` | Inferencia batch sobre el test set con el modelo fine-tuned (checkpoint-200). Carga el modelo en 4-bit sin `device_map="auto"` para evitar conflictos con Accelerate. Extrae solo la respuesta del asistente del output completo. | `data/dataset_test.json`, `models/resultados_rgpd_llama3/checkpoint-200` | `data/resultados_para_ragas.json` |
| `fine_tuning_ragas_evaluation.py` | Evalua el modelo fine-tuned con 2 metricas RAGAS: `AnswerRelevancy` y `AnswerCorrectness`. Usa `ground_truth` como contexto simulado. Embeddings: all-MiniLM-L6-v2 (safetensors, sin errores de seguridad). | `data/resultados_para_ragas.json` | `data/metricas_finetuning_resultados.csv` |

### Utilidades / Tests

| Script | Descripcion | Lee | Escribe |
|--------|-------------|-----|---------|
| `rag_pipeline.py` | Demo interactiva del RAG: conecta a `db_rgpd`, responde una pregunta predefinida citando articulos. Util para probar el sistema end-to-end. | `data/db_rgpd/` | — |
| `test_fine_tuning.py` | Test rapido del modelo fine-tuned: lanza una pregunta sobre el Articulo 17 (derecho al olvido) y muestra la respuesta en consola. | `models/resultados_rgpd_llama3/checkpoint-200` | — |
| `test_query.py` | Test rapido de ChromaDB: hace una similarity search en `db_rgpd` y muestra los 3 fragmentos mas relevantes. | `data/db_rgpd/` | — |

---

## Archivos de Datos (`data/`)

### Bases de datos vectoriales
| Archivo | Descripcion |
|---------|-------------|
| `db_rgpd/` | ChromaDB indexada con PyPDFLoader (texto sin limpiar) |
| `db_rgpd_pymupdf/` | ChromaDB indexada con PyMuPDF + ftfy (texto limpio, **version usada en produccion**) |

### Datasets intermedios
| Archivo | Producido por | Consumido por |
|---------|---------------|---------------|
| `golden_dataset_raw_limpio100.json` | `golden_dataset_generator.py` | `rag_inference_ragas.py`, `fine_tuning_rgpd.py` |
| `ragas_dataset_limpio100.json` | `rag_inference_ragas.py` | `ragas_evaluation.py` |
| `dataset_test.json` | `fine_tuning_rgpd.py` | `batch_eval.py` |
| `resultados_para_ragas.json` | `batch_eval.py` | `fine_tuning_ragas_evaluation.py` |

### Resultados de evaluacion
| Archivo | Metricas | Pipeline |
|---------|----------|----------|
| `resultados_ragas_limpio100.csv` | Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall | RAG Baseline |
| `metricas_finetuning_resultados.csv` | AnswerRelevancy, AnswerCorrectness | Fine-Tuning |
| `metricas_finetuning_ragas.csv` | (version alternativa) | Fine-Tuning |

### Versiones anteriores (desarrollo)
`golden_dataset_raw.json`, `golden_dataset_raw2.json`, `ragas_dataset.json`, `ragas_dataset_corregido.json`, `ragas_dataset_v2.json`, `ragas_dataset_v3.json`, `resultados_ragas_finales.csv`, `resultados_ragas_corregido.csv`

---

## Modelos (`models/`)

| Ruta | Descripcion |
|------|-------------|
| `models/resultados_rgpd_llama3/` | Directorio de entrenamiento SFTTrainer. Checkpoints en pasos 50, 100, 150, 200. |
| `models/resultados_rgpd_llama3/checkpoint-200` | Checkpoint final usado en `batch_eval.py` y `test_fine_tuning.py`. |
| `models/llama3_rgpd_lora/` | Adaptadores LoRA guardados con `save_pretrained()` al finalizar el entrenamiento. |

---

## Modelos y Tecnologias

| Componente | Modelo / Tecnologia |
|------------|---------------------|
| LLM base | `meta-llama/Meta-Llama-3-8B-Instruct` |
| LLM local (Ollama) | `llama3:8b` — generador RAG y juez RAGAS |
| Embeddings RAG | `BAAI/bge-m3` — alta calidad multilingue |
| Embeddings RAGAS | `sentence-transformers/all-MiniLM-L6-v2` — compatible safetensors |
| Vector DB | ChromaDB (persistente en disco) |
| Fine-tuning | QLoRA 4-bit (nf4) + PEFT + SFTTrainer (trl) |
| Evaluacion | RAGAS framework |
| Cuantizacion inferencia | BitsAndBytes 4-bit (float16) |

---

## Configuracion Inicial

### Requisitos

| Componente | RAG Pipeline | Fine-Tuning |
|---|---|---|
| Python | 3.10 – 3.12 | 3.10 – 3.12 |
| GPU VRAM | Opcional (Ollama gestiona CPU/GPU) | **Minimo 8 GB** (recomendado 16 GB) |
| PyTorch | CPU-only valido | **Con CUDA obligatorio** |
| Ollama | Requerido (`llama3:8b`) | No necesario |
| Entorno recomendado | Local | Google Colab (T4/A100) |

> **Nota**: GTX 1650 (4 GB VRAM) + `torch+cpu` **no puede ejecutar fine-tuning**.
> Usar Google Colab o servidor con GPU ≥8 GB y CUDA.

### Instalacion

```powershell
# Modelos Ollama
ollama pull llama3:8b

# Entorno Python
python -m venv venv
.\venv\Scripts\activate

# Dependencias RAG + evaluacion (funcionan con torch+cpu)
pip install "langchain==0.3.25" "langchain-community" "langchain-ollama==0.2.3"
pip install chromadb sentence-transformers ragas pandas streamlit plotly
pip install pypdf pymupdf ftfy python-dotenv

# Fine-tuning (solo en maquina con CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes datasets scikit-learn
```

> **Evitar** `langchain-chroma>=1.1.0` y `langchain-huggingface>=1.2.2`: arrastran
> `langchain-protocol` que es incompatible con Python ≤3.12.

### Variables de entorno

Crear `.env` en la raiz del proyecto:
```
HF_TOKEN=hf_xxxxxxxxxxxx
```
Requerido por `fine_tuning_rgpd.py` para descargar `meta-llama/Meta-Llama-3-8B-Instruct` desde Hugging Face.

---

## Orden de Ejecucion

### Pipeline A (RAG Baseline)
```powershell
python src/preprocessing_and_clean.py       # 1. Indexar PDF limpio
python src/golden_dataset_generator.py      # 2. Generar golden dataset (100 QA pairs)
python src/rag_inference_ragas.py           # 3. Generar respuestas RAG sobre el dataset
python src/ragas_evaluation.py              # 4. Evaluar con RAGAS (4 metricas)
```

### Pipeline B (Fine-Tuning)
```powershell
python src/fine_tuning_rgpd.py              # 1. Entrenar modelo (requiere GPU, ~horas)
python src/batch_eval.py                    # 2. Inferencia batch en test set
python src/fine_tuning_ragas_evaluation.py  # 3. Evaluar con RAGAS (2 metricas)
```

### Diagnostico y Tests
```powershell
python src/comparacion_bbdd.py              # Comparar calidad de extraccion PDF
python src/test_query.py                    # Probar busqueda vectorial en ChromaDB
python src/test_fine_tuning.py              # Probar modelo fine-tuned (una pregunta)
python src/rag_pipeline.py                  # Demo RAG interactiva
```

### Dashboard interactivo
```powershell
streamlit run src/utils/dashboard_tfm.py
```

---

## Mejoras aplicadas (rama `feature/mejoras-arquitectura`)

### Problema 1 — Answer Relevancy baja en RAG (0.71–0.77)
**Causa**: k=3 insuficiente + prompt verboso que permitia divagacion.  
**Fix** en `rag_inference_ragas.py` y `rag_pipeline.py`:
- `k` 3 → 5 (mas contexto recuperado)
- Prompt con limite de 3-4 frases y prohibicion explicita de inventar

### Problema 2 — Fine-Tuning en CPU / overfitting en GPU
**Causa**: `max_steps=200` con 70 muestras (~11 epocas efectivas), `lr=2e-4` agresiva, `max_length=1024` insuficiente para contextos RGPD concatenados.  
**Fix** en `fine_tuning_rgpd.py`:

| Parametro | Antes | Despues | Razon |
|---|---|---|---|
| `max_steps` | 200 | 120 | ~3 epocas efectivas, sin overfitting |
| `learning_rate` | 2e-4 | 1e-4 | Conservadora para 70 muestras |
| `warmup_steps` | 0 | 10 | Estabiliza arranque |
| `weight_decay` | 0 | 0.01 | Regularizacion |
| `max_length` | 1024 | 2048 | Contextos RGPD superan 1024 tokens |
| LoRA modules | q/k/v/o_proj | + gate/up/down_proj | Adaptacion completa atencion+FFN |

### Problema 3 — Evaluacion fine-tuning invalida (answer_relevancy=0.442)
**Causa A**: `fine_tuning_ragas_evaluation.py` reemplazaba `contexts` con `ground_truth` → metricas de contexto inventadas.  
**Causa B**: `batch_eval.py` lanzaba inferencia SIN contexto, aunque el modelo fue entrenado CON contexto (mismatch train/eval).  
**Causa C**: Embeddings distintos (all-MiniLM vs bge-m3) hacian la comparacion RAG vs FT injusta.  
**Fix**:
- Eliminado `contexts = [ground_truth]` → error explicito si falta el campo real
- `batch_eval.py` extrae contexto del prompt del test set y lo pasa en inferencia
- `batch_eval.py` busca el checkpoint mas reciente automaticamente (no asume `-200`)
- Embedding unificado a BAAI/bge-m3 en ambas evaluaciones

### Problema 4 — Rutas hardcodeadas en ragas_evaluation.py
**Fix**: Rutas absolutas via `Path(__file__).resolve()` — funciona desde cualquier directorio.
