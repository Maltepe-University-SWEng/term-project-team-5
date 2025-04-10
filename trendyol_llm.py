#!/usr/bin/env python3
"""
Script to interact with Trendyol-LLM-7b-chat-dpo-v1.0 model using command line arguments
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate text using Trendyol LLM model")
    parser.add_argument("prompt", nargs="?", default="Fıkra anlat", 
                      help="The prompt to send to the model")
    parser.add_argument("--system_prompt", default=None, 
                      help="Custom system prompt (optional)")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, 
                      help="Temperature for sampling (0.0-1.0)")
    args = parser.parse_args()

    model_id = "Trendyol/Trendyol-LLM-7b-chat-dpo-v1.0"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_use_double_quant=True,   
        bnb_4bit_quant_type="nf4",       
    )

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map='auto',
        quantization_config=quantization_config,
    )

    sampling_params = dict(
        do_sample=True,
        temperature=args.temperature,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=args.max_tokens, 
        return_full_text=True,
    )

    DEFAULT_SYSTEM_PROMPT = "Sen yardımcı bir asistansın ve sana verilen talimatlar doğrultusunda en iyi cevabı üretmeye çalışacaksın.\n"
    system_prompt = args.system_prompt if args.system_prompt else DEFAULT_SYSTEM_PROMPT

    TEMPLATE = (
        "[INST] {system_prompt}\n\n"
        "{instruction} [/INST]"
    )

    def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
        return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})

    def generate_output(user_query, sys_prompt=DEFAULT_SYSTEM_PROMPT):
        prompt = generate_prompt(user_query, sys_prompt)
        outputs = pipe(
            prompt,
            **sampling_params
        )
        return outputs[0]["generated_text"].split("[/INST]")[-1]

    print(f"Generating response for prompt: '{args.prompt}'")
    response = generate_output(args.prompt, system_prompt)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
