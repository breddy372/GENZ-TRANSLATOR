# Gen-Z Translator — Fine-tuned LLaMA 3.2 1B

A Gen-Z slang translator powered by a LoRA fine-tuned LLaMA 3.2 1B model, served as a FastAPI endpoint.
**The server automatically downloads and caches the model from Hugging Face Hub (`Bhargavreddy1/Llama-3.2-1B-GenZ-Translator-v1`) on startup.**

## Prerequisites

- **GPU Required**: An NVIDIA GPU with CUDA support and at least **6GB of VRAM** (e.g., GTX 1660 Ti or better). The model loads in full float16 for maximum stability and speed.
- **Python**: Python 3.10 or higher.
- **Hugging Face Account**: You need a Hugging Face account with a Read/Write Access Token.

## Installation & Setup

### 1. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Hugging Face Token
1. Go to [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens).
2. Create a new token with **Read/Write** permissions.
3. In the root directory of this project, create a `.env` file and add your token:

```env
HF_TOKEN=your_hf_read_write_token_here
MODEL_WRITER_TOKEN=your_hf_read_write_token_here
```

## Quick Start

### 1. Start the API server
```bash
# From project root
.\venv\Scripts\python.exe app.py
```

The model will be downloaded from Hugging Face on the first run.
Server starts at **http://localhost:8000**

### 2. Use the API

**Swagger UI**: http://localhost:8000/docs

**cURL**:
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "I am really tired today, I need some rest."}'
```

**Python**:
```python
import requests

response = requests.post("http://localhost:8000/translate", json={
    "text": "I'm really excited about this!",
    "max_new_tokens": 100,
    "temperature": 0.7,
})
print(response.json()["gen_z_text"])
```

## Advanced: Local Merging & CLI Inference

If you want to merge your own LoRA adapter locally instead of using the Hugging Face model:

**Save the merged model (one-time)**
```bash
cd tests
..\venv\Scripts\python.exe save_and_load_model.py
```

**CLI mode (no server)**
```bash
cd tests
..\venv\Scripts\python.exe save_and_load_model.py --interactive
```

## API Endpoints

| Method | Endpoint     | Description                     |
|--------|--------------|---------------------------------|
| POST   | `/translate` | Translate text to Gen-Z slang   |
| GET    | `/health`    | Model status and GPU info       |
| GET    | `/docs`      | Interactive Swagger UI          |

## Project Structure

```
LORA/
├── app.py                          # FastAPI server
├── requirements.txt                # Python dependencies
├── dockerfile                      # Docker build config
├── .env                            # HF_TOKEN
├── src/
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response schemas
│   └── services/
│       └── model_service.py        # Model loading (loads from HF) and inference
├── tests/
│   ├── llama3_genz_final/          # Locally merged model weights (if generated)
│   ├── results/checkpoint-378/     # LoRA adapter from training
│   ├── save_and_load_model.py      # Local merge & CLI inference script
│   └── LORAfinetuningLLAMA3.ipynb  # Training & Hugging Face upload notebook
└── data/
    └── file.7z                     # Training data
```

## Docker

Two Dockerfiles are provided:

| File | Use case |
|------|----------|
| `dockerfile` | **GPU mode** — CUDA 12.1 image, requires NVIDIA GPU passthrough |
| `dockerfile.cpu` | **CPU-only mode** — slim Python image, works anywhere, slower inference |

---

### Prerequisites

#### GPU mode (recommended)
- **Docker Desktop** with **WSL 2 backend** enabled  
  _(Settings → General → "Use the WSL 2 based engine" ✅)_
- **NVIDIA GPU** with ≥ 6 GB VRAM (e.g. GTX 1660 Ti or better)
- **NVIDIA drivers v525+** installed on the Windows host  
  Verify: `nvidia-smi` should show your GPU
- Confirm GPU passthrough into Docker works:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
  You should see your GPU listed inside the container output.

#### CPU mode
- Docker Desktop only — no GPU or special drivers needed.

---

### 1. Build the image

**GPU build (default):**
```bash
docker build -t genz-translator .
```

**CPU-only build:**
```bash
docker build -f dockerfile.cpu -t genz-translator-cpu .
```

---

### 2. Pass your Hugging Face token

The model is downloaded from Hugging Face Hub on first startup. Pass your token as an environment variable:

```bash
# GPU
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_hf_token_here \
  --name genz-api genz-translator

# CPU
docker run -p 8000:8000 \
  -e HF_TOKEN=your_hf_token_here \
  --name genz-api-cpu genz-translator-cpu
```

Or load it from your `.env` file automatically:

```bash
# GPU
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  --name genz-api genz-translator
```

---

### 3. Cache the model between runs (recommended)

On first startup the model (~2.4 GB) is downloaded from Hugging Face.  
Mount a local volume so it isn't re-downloaded on every `docker run`:

```bash
# GPU — cache model to a local folder
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -v "%cd%\hf_cache:/root/.cache/huggingface" \
  --name genz-api genz-translator

# PowerShell variant (use ${PWD} instead of %cd%)
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -v "${PWD}/hf_cache:/root/.cache/huggingface" \
  --name genz-api genz-translator
```

---

### 4. Docker Compose (easiest way to run)
Security
SOC 2 compliance
HIPAA compatibility
Audit logs
Static IP proxy
RBAC
SSO
Create a `docker-compose.yml` in the project root:

```yaml
services:
  app:
    build: .                          # use dockerfile.cpu for CPU mode
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then:

```bash
# Start (builds automatically if image doesn't exist)
docker compose up

# Start in background (detached)
docker compose up -d

# Stop
docker compose down
```

---

### 5. Verify the container is running

Wait for `✅ Model ready!` in the logs (~30–60s on first run), then:

**Health check:**
```bash
curl http://localhost:8000/health
```

**Translate:**
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "I am really tired today, I need some rest."}'
```

**Interactive Swagger UI:** http://localhost:8000/docs

---

### 6. Useful container commands

```bash
# Stream live logs
docker logs genz-api -f

# Check container status
docker ps -a

# Open a shell inside the running container
docker exec -it genz-api bash

# Confirm GPU is visible inside the container
docker exec -it genz-api nvidia-smi

# Stop the container
docker stop genz-api

# Remove the container (keeps the image)
docker rm genz-api

# Stop AND remove in one step
docker rm -f genz-api
```

---

### 7. Rebuild after code changes

```bash
# Rebuild image (no cache, picks up all file changes)
docker build --no-cache -t genz-translator .

# Then re-run
docker run --gpus all -p 8000:8000 --env-file .env --name genz-api genz-translator
```

---

### 8. Image & disk management

```bash
# List all images
docker images

# Remove the built image
docker rmi genz-translator

# Remove ALL stopped containers and dangling images (reclaim disk space)
docker system prune

# Nuclear option — remove everything including volumes and cached layers
docker system prune -a --volumes
```

---

### 9. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Found no NVIDIA driver` | Container started without `--gpus all` | Add `--gpus all` flag to `docker run` |
| `nvidia-smi` fails in container | GPU passthrough not configured | Enable WSL 2 backend in Docker Desktop; verify `nvidia-smi` works in WSL |
| `docker: Error response: unknown flag: --gpus` | Old Docker version | Update Docker Desktop to v20.10+ |
| Model not downloading | Missing or invalid HF token | Pass `--env-file .env` or `-e HF_TOKEN=...` |
| Port already in use | Another process on port 8000 | Change to `-p 8001:8000` and access via port 8001 |
| Container exits immediately | Startup error (model load failed) | Run `docker logs genz-api` to see the error |

