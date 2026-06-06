"""
modal_app.py — Deploy the Gen-Z Translator FastAPI API on Modal.com

Deploy:
    python -m modal deploy modal_app.py

Serve (ephemeral, for testing):
    python -m modal serve modal_app.py

The FastAPI app is served via Modal's @asgi_app web function, giving it
a stable public HTTPS URL. The LLaMA model is loaded from HuggingFace and
cached in a Modal Volume so subsequent cold-starts are fast.

Requirements (install locally before deploying):
    pip install modal huggingface_hub
    python -m modal token new   # authenticates your CLI once
"""

import modal

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Persistent volume — caches downloaded HuggingFace model weights.
#     Without this every cold-start re-downloads ~2.4 GB.
# ──────────────────────────────────────────────────────────────────────────────
volume = modal.Volume.from_name("genz-translator-cache", create_if_missing=True)
CACHE_DIR = "/cache/huggingface"

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Container image — pip-installs dependencies, then copies local source
#     files (app.py + src/) into /root/ so they are importable at runtime.
#     Modal v1+ uses image.add_local_* instead of the removed modal.Mount.
# ──────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
        add_python="3.11",
    )
    .pip_install(
        "accelerate>=0.28.0",
        "transformers>=4.40.0,<4.48.0",
        "tokenizers>=0.19.0",
        "tiktoken",
        "sentencepiece",
        "python-dotenv",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "huggingface_hub",
    )
    .env({"HF_HOME": CACHE_DIR, "FORCE_REBUILD_CACHE": "1"})
    # Copy local source files into the container image at /root/
    # so `from app import app` and `from src.services...` resolve correctly.
    .add_local_file("app.py", remote_path="/root/app.py")
    .add_local_dir("src", remote_path="/root/src")
)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Modal App
# ──────────────────────────────────────────────────────────────────────────────
app = modal.App("genz-translator", image=image)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  ASGI Web Function — wraps the existing FastAPI app directly.
#
#     @modal.asgi_app() turns the function's return value (a FastAPI/ASGI app)
#     into a persistent HTTPS endpoint. Modal auto-scales containers and keeps
#     one warm (min_containers=1) to avoid cold-start latency on first request.
# ──────────────────────────────────────────────────────────────────────────────
@app.function(
    # GPU: L4 (24 GB VRAM) — great price/perf for a 1B-param fp16 model.
    # Alternatives: "T4" (cheaper), "A10G" (faster throughput).
    gpu="L4",

    # RAM: transformers + model weights need ~6 GB minimum.
    memory=8192,  # MB

    # Timeout: allow up to 5 minutes per request for long generations.
    timeout=300,

    # Keep one container always warm — avoids the ~60s model-load cold-start.
    min_containers=1,

    # Mount the persistent volume at the HuggingFace cache directory.
    volumes={CACHE_DIR: volume},

    # Expose HF_TOKEN from a Modal Secret named "huggingface-secret".
    # Create it once with:
    #   python -m modal secret create huggingface-secret HF_TOKEN=<your_token>
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def fastapi_app():
    """
    Entry point for the Modal web function.

    Modal calls this once per container startup. Returns the FastAPI ASGI app.
    The lifespan handler in app.py (which loads the model) runs automatically
    when the ASGI server starts.
    """
    import os
    import sys

    # Ensure /root is on sys.path so `import app` and `from src...` work.
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    # Tell HuggingFace to use the mounted volume for caching.
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)

    # Import and return the FastAPI app from app.py.
    from app import app as fastapi_application
    return fastapi_application


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Pre-warm helper — downloads model weights to the Modal Volume.
#     Run once manually: python -m modal run modal_app.py::download_model
# ──────────────────────────────────────────────────────────────────────────────
@app.function(
    gpu="L4",
    memory=8192,
    timeout=600,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model():
    """
    Pre-downloads model weights into the Modal Volume.
    Avoids paying the download cost on every cold-start.
    """
    import os
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = CACHE_DIR

    MODEL_ID = "Bhargavreddy1/Llama-3.2-1B-GenZ-Translator-v1"
    print(f"Downloading {MODEL_ID} → {CACHE_DIR} ...")
    snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR)

    volume.commit()
    print("✅ Model downloaded and cached in Modal Volume.")
