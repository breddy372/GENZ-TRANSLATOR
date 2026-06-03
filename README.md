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

> **Requirements:** Docker Desktop with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed for GPU support.

### 1. Build the image
```bash
docker build -t genz-translator .
```

### 2. Run the container

**With GPU (recommended):**
```bash
docker run --gpus all -p 8000:8000 --name genz-api genz-translator
```

**CPU only (slower inference):**
```bash
docker run -p 8000:8000 --name genz-api genz-translator
```

Wait for the log line `✅ Model ready!` before sending requests (~30–60s on first run).

### 3. Test the container

**Health check:**
```bash
curl http://localhost:8000/health
```

**Translate:**
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "I am really tired today."}'
```

**Interactive Swagger UI:** http://localhost:8000/docs

### 4. View logs
```bash
docker logs genz-api -f
```

### 5. Stop and remove
```bash
docker stop genz-api
docker rm genz-api
```
