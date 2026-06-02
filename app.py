"""
Gen-Z Translator API — FastAPI server for the fine-tuned LLaMA 3.2 1B model.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /translate  — Translate normal English → Gen-Z slang
    GET  /health     — Check model status and GPU info
    GET  /docs       — Interactive Swagger UI (auto-generated)

The model loads ONCE at server startup (lifespan event), then stays in GPU
memory for the lifetime of the process. Every /translate request reuses the
same model — no reloading, instant replies after the first warm-up.
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.services.model_service import model_service
from src.models.schemas import (
    TranslateRequest,
    TranslateResponse,
    HealthResponse,
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


# ──────────────────────────────────────────────
# Lifespan: load model on startup, free on shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    - Startup:  loads the model into GPU memory (one-time, ~30-60s).
    - Shutdown: unloads the model and frees GPU memory.
    """
    logger.info("🚀 Starting Gen-Z Translator API...")
    logger.info("📦 Loading model into GPU (this takes ~30-60s on first run)...")

    try:
        model_service.load()
        gpu_info = model_service.get_gpu_info()
        logger.info(f"✅ Model ready! GPU: {gpu_info.get('gpu', 'N/A')} | "
                     f"VRAM: {gpu_info.get('vram_used_mb', '?')} MB used")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("Run 'python tests/save_and_load_model.py' first to create the merged model.")
        raise

    yield  # Server is now running and accepting requests

    logger.info("🛑 Shutting down — unloading model...")
    model_service.unload()
    logger.info("✅ Cleanup complete.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Gen-Z Translator API",
    description=(
        "Translate normal English into Gen-Z slang using a fine-tuned "
        "LLaMA 3.2 1B model (LoRA fine-tuned on genz-slang-pairs-1k dataset). "
        "The model loads once at startup and stays in GPU memory for instant replies."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current status of the model and GPU.",
    tags=["System"],
)
async def health():
    """Check if the model is loaded and return GPU/VRAM info."""
    gpu_info = model_service.get_gpu_info()

    return HealthResponse(
        status="healthy" if model_service.is_loaded else "loading",
        model_loaded=model_service.is_loaded,
        model_path=model_service.model_path,
        gpu=gpu_info["gpu"],
        vram_used_mb=gpu_info["vram_used_mb"],
        vram_total_mb=gpu_info["vram_total_mb"],
    )


@app.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate text to Gen-Z slang",
    description=(
        "Submit normal English text and receive a Gen-Z slang translation. "
        "The model is already loaded in memory — typical response time is 2-10 seconds."
    ),
    tags=["Translation"],
)
async def translate(request: TranslateRequest):
    """
    Core translation endpoint.
    Runs the fine-tuned LLaMA model on the provided text.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Try again in a minute.",
        )

    start = time.time()

    try:
        gen_z_text = model_service.infer(
            user_message=request.text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            system_prompt=request.system_prompt,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )

    elapsed = time.time() - start
    approx_tokens = len(gen_z_text.split())

    logger.info(
        f"Translated ({elapsed:.2f}s, ~{approx_tokens} tokens): "
        f"'{request.text[:50]}...' → '{gen_z_text[:50]}...'"
    )

    return TranslateResponse(
        input_text=request.text,
        gen_z_text=gen_z_text,
        model="LLaMA-3.2-1B-GenZ-LoRA",
        tokens_generated=approx_tokens,
    )


# ──────────────────────────────────────────────
# Run directly: python app.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Disable reload — model loading is expensive
        log_level="info",
    )
