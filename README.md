# Gen-Z Translator — Fine-tuned LLaMA 3.2 1B

A Gen-Z slang translator powered by a LoRA fine-tuned LLaMA 3.2 1B model, served as a FastAPI endpoint. 
**The server automatically downloads and caches the model from Hugging Face Hub (`Bhargavreddy1/Llama-3.2-1B-GenZ-Translator-v1`) on startup.**

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
