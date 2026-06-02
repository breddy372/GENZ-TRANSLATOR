# Gen-Z Translator — Fine-tuned LLaMA 3.2 1B

A Gen-Z slang translator powered by a LoRA fine-tuned LLaMA 3.2 1B model,
served as a FastAPI endpoint.

## Quick Start

### 1. Save the merged model (one-time)
```bash
cd tests
..\venv\Scripts\python.exe save_and_load_model.py
```

### 2. Start the API server
```bash
# From project root
.\venv\Scripts\python.exe app.py
```

Server starts at **http://localhost:8000**

### 3. Use the API

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

### 4. CLI mode (no server)
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
│       └── model_service.py        # Model loading and inference logic
├── tests/
│   ├── llama3_genz_final/          # Merged model weights (~2.4 GB)
│   ├── results/checkpoint-378/     # LoRA adapter from training
│   ├── save_and_load_model.py      # Save/load/CLI inference script
│   └── LORAfinetuningLLAMA3.ipynb  # Training notebook
└── data/
    └── file.7z                     # Training data
```
