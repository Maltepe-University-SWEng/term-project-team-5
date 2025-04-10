#!/usr/bin/env python3
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py \"Your prompt here\"")
        return
    
    prompt = " ".join(sys.argv[1:])
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        "TURKCELL/Turkcell-LLM-7b-v1",
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "TURKCELL/Turkcell-LLM-7b-v1",
        use_fast=True
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    
    eos_token = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
    
    print("Generating response...\n")
    with torch.no_grad():
        generated_ids = model.generate(
            encodeds.to(device if device == "cuda" else model.device),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            eos_token_id=eos_token,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("-" * 40)
    print(decoded[0])
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
