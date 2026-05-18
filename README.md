# ⚖️ Asistente RAG para Consultas Legales (RGPD)

Este proyecto es el núcleo práctico de mi TFM. Utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** para permitir consultas en lenguaje natural sobre el Reglamento General de Protección de Datos (RGPD), procesando todo de forma **100% local** para garantizar la privacidad.

---

## 🚀 Guía de Configuración Inicial (Solo una vez)

Estas tareas preparan el "cerebro" y el entorno del sistema. Una vez completadas, no es necesario repetirlas.

1.  **Instalar Ollama:** Descargar desde [ollama.com](https://ollama.com) e instalar en Windows.
2.  **Descargar Llama 3:** Abrir una terminal y ejecutar:
    ```powershell
    ollama pull llama3:8b
    ```
3.  **Configurar Entorno Python:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    pip install langchain-ollama langchain-huggingface langchain-chroma langchain-community pypdf
    ```
4.  **Generar Base de Datos Vectorial:**
    Ejecutar el script de preprocesamiento para convertir el PDF en vectores:
    ```powershell
    python preprocessing.py
    ```
    *Esto creará la carpeta `db_rgpd/` con los datos ya indexados.*

---

## 🏃 Ejecución Diaria (Flujo de Trabajo)

Sigue estos pasos cada vez que quieras realizar consultas al sistema:

### Paso 1: Activar el Servidor de IA
Asegúrate de que el programa **Ollama** esté abierto.
* Busca el icono de la **llama** en la barra de tareas (junto al reloj).
* Si no está, abre "Ollama" desde el menú Inicio.

### Paso 2: Activar el Entorno Virtual
Abre la terminal en VS Code y escribe:
```powershell
.\venv\Scripts\activate