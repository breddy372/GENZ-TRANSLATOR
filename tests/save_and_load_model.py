"""
save_and_load_model.py
======================
Saves the fine-tuned LoRA model (properly merged) and loads it for inference.
Run save_merged_model() ONCE. After that, load_and_infer() works offline forever.

Paths:
  - LoRA adapter (~3.4 MB)    : tests/results/checkpoint-378/
  - Final merged model (~1 GB): tests/llama3_genz_final/

WHY lora_adapter_temp/ doesn't work as a merged model:
  It was saved while the PEFT wrapper was still attached — it contains
  lora_A/lora_B/base_layer keys instead of plain weights.

WHY we load in float16 (not 4-bit) for merging:
  merge_and_unload() does math on weights. 4-bit stores weights as uint8 (Byte),
  which can't do float math → RuntimeError: "normal_kernel_cuda" not implemented for 'Byte'

HOW inference works efficiently:
  The model is loaded ONCE into a module-level cache (_model_cache).
  Every subsequent call to load_and_infer() reuses the cached model/tokenizer/pipe
  — no reloading, no waiting, instant replies.
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

from pathlib import Path

# Resolve paths dynamically relative to this script's directory (tests/)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH    = str(SCRIPT_DIR / "results" / "checkpoint-378")
FINAL_SAVE_PATH = str(SCRIPT_DIR / "llama3_genz_final")       # Where the merged model is saved

# ──────────────────────────────────────────────
# MODULE-LEVEL CACHE — model loaded once, reused forever
# ──────────────────────────────────────────────
_model_cache = {
    "pipe": None,        # The text-generation pipeline
    "tokenizer": None,   # The tokenizer
}


# ══════════════════════════════════════════════
# STEP 1 — SAVE  (run once, skip if already done)
# ══════════════════════════════════════════════

def save_merged_model():
    """
    Merges base model + LoRA adapter and saves to FINAL_SAVE_PATH.
    Only needs to run ONCE. After that, call load_and_infer() directly.

    Key points:
    - Base model is loaded in float16 (NO 4-bit), because merge_and_unload()
      requires real float tensors — not quantized uint8 bytes.
    - LLaMA 3.2 1B in float16 = ~2 GB VRAM, well within GTX 1660 Ti (6 GB).
    - After merging, the saved model is a normal HuggingFace model you can
      reload anytime with 4-bit quantization for cheap inference.
    """
    load_dotenv(override=True)

    # ── Skip if already saved ──
    if os.path.exists(os.path.join(FINAL_SAVE_PATH, "config.json")):
        print(f"✅ Merged model already exists at '{FINAL_SAVE_PATH}'. Skipping save.")
        print("   Call load_and_infer() to run inference.")
        return

    print("🔄 Step 1/4: Loading base model in float16 (no quantization)...")
    print("   Reason: merge_and_unload() requires float tensors, not 4-bit uint8.")
    print(f"   Memory: LLaMA 3.2 1B @ float16 ≈ 2 GB VRAM")

    # Load base model WITHOUT quantization on CPU so merge_and_unload() can do math safely
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
    )

    print(f"✅ Step 2/4: Base model loaded. Attaching LoRA from '{ADAPTER_PATH}'...")

    # Attach the trained LoRA adapter on top of the base model (will default to CPU since base is on CPU)
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("🔀 Step 3/4: Merging LoRA weights into base model...")
    # merge_and_unload() fuses LoRA (A·B * scale) into the base weights
    # and returns a plain LlamaForCausalLM — no adapter wrappers left
    merged_model = peft_model.merge_and_unload()

    print(f"💾 Step 4/4: Saving merged model + tokenizer to '{FINAL_SAVE_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    os.makedirs(FINAL_SAVE_PATH, exist_ok=True)
    merged_model.save_pretrained(FINAL_SAVE_PATH)
    tokenizer.save_pretrained(FINAL_SAVE_PATH)

    print(f"\n✅ SUCCESS! Merged model saved to '{FINAL_SAVE_PATH}'.")
    print("   You NEVER need to fine-tune again — just call load_and_infer().")

    # Free GPU memory
    del merged_model, peft_model, base_model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════
# STEP 2 — LOAD MODEL  (called automatically on first inference)
# ══════════════════════════════════════════════

def _load_model_into_cache():
    """
    Loads the saved merged model into the module-level _model_cache.
    This only runs ONCE per Python session. All subsequent calls to
    load_and_infer() reuse the already-loaded model — no disk reads,
    no GPU re-allocation, instant replies.
    """
    if _model_cache["pipe"] is not None:
        return  # Already loaded, nothing to do

    if not os.path.exists(os.path.join(FINAL_SAVE_PATH, "config.json")):
        raise FileNotFoundError(
            f"Merged model not found at '{FINAL_SAVE_PATH}'.\n"
            f"Run save_merged_model() first!"
        )

    print(f"📦 Loading fine-tuned model from '{FINAL_SAVE_PATH}' (first call only)...")

    # Load in full float16 (no quantization) and move to CUDA manually to avoid Windows bitsandbytes/device_map allocation crashes
    model = AutoModelForCausalLM.from_pretrained(
        FINAL_SAVE_PATH,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.config.use_cache = True  # Faster generation via KV cache

    tokenizer = AutoTokenizer.from_pretrained(FINAL_SAVE_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Store in cache — stays alive for the entire Python session
    _model_cache["pipe"]      = pipe
    _model_cache["tokenizer"] = tokenizer

    print("✅ Model loaded and cached. All future calls will be instant.\n")


# ══════════════════════════════════════════════
# STEP 3 — INFER  (fast after first call)
# ══════════════════════════════════════════════

def load_and_infer(user_message: str, max_new_tokens: int = 150):
    """
    Translates normal English to Gen-Z slang using the fine-tuned model.

    - FIRST CALL: loads model from disk into GPU (~30-60 sec, one-time cost).
    - ALL SUBSEQUENT CALLS: uses cached model — replies in seconds.

    Args:
        user_message   : Normal English text to translate.
        max_new_tokens : Maximum tokens to generate.

    Returns:
        str: The Gen-Z slang translation.
    """
    # Load model on first call, reuse cached on all subsequent calls
    _load_model_into_cache()

    pipe      = _model_cache["pipe"]
    tokenizer = _model_cache["tokenizer"]

    # Use the same chat template as training
    messages = [
        {"role": "system", "content": "You are a Gen-Z translator. Convert normal English into Gen-Z slang."},
        {"role": "user",   "content": user_message},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result     = pipe(input_text, max_new_tokens=max_new_tokens)
    full_output = result[0]["generated_text"]

    # Extract only the assistant reply (strip the prompt prefix)
    if "<|start_header_id|>assistant<|end_header_id|>" in full_output:
        response = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        response = response.replace("<|eot_id|>", "").strip()
    else:
        response = full_output

    return response


def unload_model():
    """
    Explicitly frees the model from GPU/RAM.
    Call this if you need the VRAM back for something else.
    Next load_and_infer() call will reload from disk.
    """
    global _model_cache
    if _model_cache["pipe"] is not None:
        del _model_cache["pipe"]
        del _model_cache["tokenizer"]
        _model_cache = {"pipe": None, "tokenizer": None}
        torch.cuda.empty_cache()
        print("🗑️  Model unloaded from memory.")
    else:
        print("ℹ️  Model was not loaded.")


# ══════════════════════════════════════════════
# MAIN — Run from command line
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Save the merged model (skips automatically if already done)
    save_merged_model()

    # Demo: test with a few examples
    # Note: model loads ONCE on the first call, then all replies are instant
    test_inputs = [
        "I'm really tired today, I need some rest.",
        "Did you see that movie? It was really good!",
        "I don't want to go to work today.",
    ]

    print("\n" + "═" * 60)
    print("  GEN-Z TRANSLATOR — Fine-tuned LLaMA 3.2 1B")
    print("  (model loads once, then all replies are instant)")
    print("═" * 60)

    for text in test_inputs:
        print(f"\n🧑 Input : {text}")
        reply = load_and_infer(text, max_new_tokens=80)
        print(f"🤙 Gen-Z : {reply}")
        print("-" * 60)

    # Interactive mode: python save_and_load_model.py --interactive
    if "--interactive" in sys.argv:
        print("\n💬 Interactive Gen-Z Translator (type 'quit' to exit)")
        print("   Model is already loaded — replies are instant!")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("✌️ Later!")
                break
            reply = load_and_infer(user_input)
            print(f"Gen-Z: {reply}")
