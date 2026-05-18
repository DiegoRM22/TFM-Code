import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_path = "./resultados_rgpd_llama3/checkpoint-200" 

print("1. Cargando configuración técnica (Sin Offload)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)

print("2. Cargando modelo base directamente en VRAM...")
# ELIMINAMOS device_map="auto" para que accelerate no intervenga
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    # No usamos low_cpu_mem_usage ni device_map
    trust_remote_code=True
)

print("3. Cargando adaptador LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4. Configuración de la consulta
pregunta = "¿Qué dice el RGPD sobre el derecho al olvido?"

prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEres un asistente legal experto en el RGPD. Responde de forma técnica citando el Artículo 17.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{pregunta}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# Aseguramos que los inputs vayan al mismo dispositivo que el modelo cargado
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n4. Generando respuesta técnica...\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=300,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
respuesta_limpia = respuesta_completa.split("assistant")[-1].strip()

print("-" * 50)
print(respuesta_limpia)
print("-" * 50)