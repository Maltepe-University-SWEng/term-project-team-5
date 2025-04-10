from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig

model_id = "Trendyol/Trendyol-LLM-7b-chat-dpo-v1.0"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True,   
    bnb_4bit_quant_type="nf4",       
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map='auto',
    quantization_config=quantization_config,
)

sampling_params = dict(
    do_sample=True,
    temperature=0.3,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=1024, 
    return_full_text=True,
)

DEFAULT_SYSTEM_PROMPT = "Sen yardımcı bir asistansın ve sana verilen talimatlar doğrultusunda en iyi cevabı üretmeye çalışacaksın.\n"
TEMPLATE = (
    "[INST] {system_prompt}\n\n"
    "{instruction} [/INST]"
)

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Generate prompt using template"""
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})

def generate_output(user_query, sys_prompt=DEFAULT_SYSTEM_PROMPT):
    """Generate model output for the given query"""
    prompt = generate_prompt(user_query, sys_prompt)
    outputs = pipe(
        prompt,
        **sampling_params
    )
    return outputs[0]["generated_text"].split("[/INST]")[-1]

if __name__ == "__main__":
    user_query = "Fıkra anlat"
    response = generate_output(user_query)
    print(response)
