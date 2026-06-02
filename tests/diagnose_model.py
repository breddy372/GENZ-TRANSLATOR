"""
Diagnostic script to identify why LLaMA 3.2 produces garbage output.
Tests: float16 (no quantization) vs 4-bit quantization.
Run this in your venv: python tests/diagnose_model.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

import torch
print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"Python:    {sys.version}")
print(f"Torch:     {torch.__version__}")
print(f"CUDA:      {torch.cuda.is_available()}")
print(f"GPU:       {torch.cuda.get_device_name(0)}")
print(f"Capability:{torch.cuda.get_device_capability()}")
print(f"VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import bitsandbytes as bnb
print(f"bnb:       {bnb.__version__}")

# Check bitsandbytes CUDA setup
print("\n" + "=" * 60)
print("TEST 1: bitsandbytes CUDA kernel check")
print("=" * 60)
try:
    # This tests if bnb can actually use its CUDA kernels
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    result = torch.matmul(a, b)
    print(f"  Basic CUDA matmul: OK (result mean={result.mean().item():.4f})")
except Exception as e:
    print(f"  CUDA matmul FAILED: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 2: Load model in float16 WITHOUT quantization
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Load model in FLOAT16 (no quantization)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

try:
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    print("  Model loaded in float16 successfully")

    # Check dtypes
    dtypes = set(p.dtype for p in model_fp16.parameters())
    print(f"  Parameter dtypes: {dtypes}")

    # Generate
    messages = [
        {"role": "system", "content": "You are a Gen-Z translator."},
        {"role": "user", "content": "How are you"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    pipe_fp16 = pipeline(
        "text-generation",
        model=model_fp16,
        tokenizer=tokenizer,
        max_new_tokens=50,
    )
    result = pipe_fp16(input_text)
    output = result[0]["generated_text"]
    # Extract just the assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in output:
        response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        response = output

    print(f"\n  FLOAT16 OUTPUT:\n  {response[:300]}")

    # Cleanup to free VRAM
    del model_fp16, pipe_fp16
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  FLOAT16 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# TEST 3: Load model WITH 4-bit quantization
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Load model in 4-BIT NF4 (bitsandbytes)")
print("=" * 60)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

try:
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    print("  Model loaded in 4-bit successfully")

    # Check for bfloat16 params (these would be BAD on GTX 1660 Ti)
    dtype_counts = {}
    for name, param in model_4bit.named_parameters():
        dt = str(param.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    print(f"  Parameter dtype distribution: {dtype_counts}")

    # Check if any bfloat16 exists
    has_bf16 = any("bfloat16" in str(p.dtype) for p in model_4bit.parameters())
    if has_bf16:
        print("  ⚠️  WARNING: bfloat16 params detected! GTX 1660 Ti does NOT support bf16!")
        print("  ⚠️  This is likely causing your garbage output!")
    else:
        print("  ✅ No bfloat16 params found")

    # Generate
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

    print(f"\n  4-BIT OUTPUT:\n  {response[:300]}")

    del model_4bit, pipe_4bit
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  4-BIT FAILED: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print("""
If TEST 2 (float16) works but TEST 3 (4-bit) produces garbage:
  → bitsandbytes quantization is broken on your system.
  → Fix: Use float16 or 8-bit quantization instead.

If BOTH tests produce garbage:
  → There may be a deeper CUDA/driver issue.

If BOTH tests work fine:
  → The issue is in your training, not model loading.
""")
