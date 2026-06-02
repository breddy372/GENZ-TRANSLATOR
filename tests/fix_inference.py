"""
Post-training inference fix for LoRA + 4-bit quantization.
Tests multiple approaches to identify why generation produces garbage.
Run this as cells in your notebook AFTER training completes.
"""

# =============================================================
# CELL 1: Check LoRA weight statistics
# =============================================================
print("=" * 60)
print("LoRA Weight Statistics")
print("=" * 60)
import torch

lora_params = {n: p for n, p in model.named_parameters() if "lora" in n.lower()}
print(f"Total LoRA parameters: {len(lora_params)}")
for name, param in list(lora_params.items())[:6]:  # show first 6
    print(f"  {name}: mean={param.data.float().mean():.6f}, std={param.data.float().std():.6f}, "
          f"min={param.data.float().min():.6f}, max={param.data.float().max():.6f}, dtype={param.dtype}")

# Check for NaN/Inf
has_nan = any(torch.isnan(p).any() for p in lora_params.values())
has_inf = any(torch.isinf(p).any() for p in lora_params.values())
print(f"\nNaN in weights: {has_nan}")
print(f"Inf in weights: {has_inf}")

if has_nan or has_inf:
    print("FOUND NaN/Inf! This is why output is garbage!")


# =============================================================
# CELL 2: Test manual generation (bypass pipeline)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Manual generation (no pipeline)")
print("=" * 60)

model.eval()
model.config.use_cache = True

messages = [
    {"role": "system", "content": "You are a Gen-Z translator. Convert normal English into Gen-Z slang."},
    {"role": "user", "content": "How are you"},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Manual generate output: {response[:200]}")

has_garbage = any(w in response[:100] for w in ["import", "def", "Question", "Tags"])
print(f"Is garbage: {has_garbage}")


# =============================================================
# CELL 3: Save adapter, reload on float16 base, test
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Save adapter -> Reload on float16 base model")
print("=" * 60)

# Save the LoRA adapter
adapter_path = "./lora_adapter_temp"
model.save_pretrained(adapter_path)
print(f"Adapter saved to {adapter_path}")

# Free VRAM
del model
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM freed. Available: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

# Reload base model in float16 (NO quantization)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

model_name = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading base model in float16...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

# Load LoRA adapter onto float16 base
print("Loading LoRA adapter...")
model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
model_with_lora.eval()

# Merge LoRA into base weights
print("Merging LoRA weights...")
merged_model = model_with_lora.merge_and_unload()

# Test generation
print("Generating with merged float16 model...")
tokenizer_fresh = AutoTokenizer.from_pretrained(model_name)
tokenizer_fresh.pad_token = tokenizer_fresh.eos_token

messages = [
    {"role": "system", "content": "You are a Gen-Z translator. Convert normal English into Gen-Z slang."},
    {"role": "user", "content": "How are you"},
]
input_text = tokenizer_fresh.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer_fresh(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

response = tokenizer_fresh.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nFLOAT16 MERGED OUTPUT:\n{response[:300]}")

has_garbage = any(w in response[:100] for w in ["import", "def", "Question", "Tags"])
if has_garbage:
    print("\nSTILL GARBAGE - the LoRA weights themselves are bad")
    print("The training learned wrong patterns. Need to adjust hyperparameters.")
else:
    print("\nSUCCESS! The LoRA training is fine!")
    print("The issue was 4-bit quantization during inference.")
    print("Use this float16+merge approach for inference going forward.")

# Additional test prompts
print("\n--- Additional test prompts ---")
test_prompts = [
    "I'm really tired today",
    "That movie was so good",
    "Let's hang out this weekend",
]
for prompt in test_prompts:
    messages = [
        {"role": "system", "content": "You are a Gen-Z translator. Convert normal English into Gen-Z slang."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer_fresh.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_fresh(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = merged_model.generate(**inputs, max_new_tokens=60, do_sample=False)
    response = tokenizer_fresh.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  '{prompt}' -> {response[:150]}")
