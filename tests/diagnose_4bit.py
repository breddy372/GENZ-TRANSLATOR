"""Test 4-bit NF4 quantization only."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv(override=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

print("Loading model in 4-bit...")
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
)
print("Model loaded successfully")

# Check dtypes
dtype_counts = {}
for name, param in model_4bit.named_parameters():
    dt = str(param.dtype)
    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
print(f"Parameter dtypes: {dtype_counts}")

has_bf16 = any("bfloat16" in str(p.dtype) for p in model_4bit.parameters())
if has_bf16:
    print("WARNING: bfloat16 params detected! GTX 1660 Ti does NOT support bf16!")
else:
    print("OK: No bfloat16 params found")

# Generate text
print("\nGenerating text with 4-bit model...")
messages = [
    {"role": "system", "content": "You are a Gen-Z translator."},
    {"role": "user", "content": "How are you"},
]
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
pipe_4bit = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    max_new_tokens=50,
)
result = pipe_4bit(input_text)
output = result[0]["generated_text"]

if "<|start_header_id|>assistant<|end_header_id|>" in output:
    response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
else:
    response = output

print(f"\n4-BIT OUTPUT:\n{response[:300]}")

# Verdict
print("\n" + "=" * 60)
has_garbage = any(word in response[:100] for word in ["import", "def", "Question", "Tags"])
if has_garbage:
    print("RESULT: 4-bit quantization is CORRUPTING the model output!")
    print("FIX: Use float16 instead of 4-bit quantization for training.")
else:
    print("RESULT: 4-bit quantization output looks OK.")
    print("The issue is likely in your LoRA training, not quantization.")
