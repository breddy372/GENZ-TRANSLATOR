"""
Model service — loads the fine-tuned LLaMA model once and serves inference.

Used by:
  - app.py (FastAPI server)
  - tests/save_and_load_model.py (CLI tool)

Model path: tests/llama3_genz_final/  (merged model, ~2.4 GB on disk)
Inference:  4-bit quantized via bitsandbytes (~0.7 GB VRAM)
"""

import os
import time
import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Resolve model path relative to project root
# ──────────────────────────────────────────────
MODEL_ID = "Bhargavreddy1/Llama-3.2-1B-GenZ-Translator-v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a Gen-Z translator. Convert normal English into Gen-Z slang."
)


class ModelService:
    """
    Singleton-style model service.
    Call load() once at startup, then infer() for every request.
    """

    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.model = None
        self.model_path = MODEL_ID
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """
        Load the merged model from Hugging Face Hub into GPU memory.
        This should be called ONCE at server startup.
        """
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return

        logger.info(f"Loading model '{self.model_path}' from Hugging Face Hub...")
        start = time.time()

        # Load in full float16 (no quantization). Since LLaMA 3.2 1B is tiny (~2.4 GB in fp16),
        # it fits beautifully on your GTX 1660 Ti (6 GB) with ~3 GB to spare.
        # We load on CPU and move to CUDA to prevent `accelerate` device_map DLL crashes on Windows.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.model.config.use_cache = True  # KV cache for faster generation

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self._loaded = True
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s ✅")

    def infer(
        self,
        user_message: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ) -> str:
        """
        Run inference on the loaded model.

        Args:
            user_message:    The text to translate.
            max_new_tokens:  Max tokens in the reply.
            temperature:     Sampling temperature (0 = greedy, >1 = creative).
            top_p:           Nucleus sampling threshold.
            system_prompt:   Custom system prompt (optional).

        Returns:
            The model's generated response text.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_message},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get stop token IDs to prevent LLaMA from rambling after the assistant turn
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [self.tokenizer.eos_token_id]
        if eot_id is not None:
            eos_ids.append(eot_id)

        result = self.pipe(
            input_text,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),  # avoid division by zero
            top_p=top_p,
            eos_token_id=eos_ids,
            return_full_text=False,
        )

        full_output = result[0]["generated_text"]
        
        # Strip any trailing special tokens (e.g. <|eot_id|>)
        response = full_output.split("<|")[0].strip()

        return response

    def unload(self):
        """Free GPU memory. Next call to load() will reload from disk."""
        if self._loaded:
            del self.pipe, self.model, self.tokenizer
            self.pipe = None
            self.model = None
            self.tokenizer = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("Model unloaded from GPU memory.")

    def get_gpu_info(self) -> dict:
        """Return GPU name and VRAM usage, if available."""
        info = {"gpu": None, "vram_used_mb": None, "vram_total_mb": None}
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_used_mb"] = round(
                torch.cuda.memory_allocated(0) / 1024 / 1024, 1
            )
            info["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1
            )
        return info


# ──────────────────────────────────────────────
# Module-level singleton instance
# ──────────────────────────────────────────────
model_service = ModelService()
