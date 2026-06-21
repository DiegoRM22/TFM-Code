import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

checkpoint_dir = MODELS_DIR / "fine_tuning" / "colab_checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {checkpoint_dir}")
adapter_path = str(checkpoints[-1])
print(f"Usando checkpoint: {adapter_path}")

print("1. Cargando configuración técnica...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

print("2. Cargando modelo base en VRAM...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

print("3. Cargando adaptador LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

pregunta = "¿Qué dice el RGPD sobre el derecho al olvido?"

prompt = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"Eres un asistente legal experto en el RGPD. Responde de forma técnica citando el Artículo 17."
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{pregunta}"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n4. Generando respuesta...\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
    )

respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()
print("-" * 50)
print(respuesta)
print("-" * 50)
