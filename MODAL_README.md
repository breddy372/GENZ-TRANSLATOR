# Gen-Z Translator — Modal Serverless Deployment

This document specifically covers how to deploy the fine-tuned **LLaMA 3.2 1B Gen-Z Translator** as a serverless FastAPI endpoint using [Modal.com](https://modal.com). 

By deploying on Modal, the application runs on a cloud GPU (NVIDIA L4) and scales to zero when not in use, making it incredibly cost-effective compared to running a dedicated GPU server 24/7.

## 🌟 Architecture Overview

- **Cloud Provider**: Modal
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Framework**: FastAPI
- **Model**: `Bhargavreddy1/Llama-3.2-1B-GenZ-Translator-v1`
- **Caching**: Modal Persistent Volumes (avoids downloading the 2.4GB model on every cold start)

---

## 🛠️ Prerequisites

1. **Python Environment**: Ensure you have Python installed and your virtual environment activated.
2. **Modal Account**: Sign up at [Modal.com](https://modal.com).
3. **Modal Python SDK**: Install the Modal library.
   ```bash
   pip install modal
   ```
4. **Authenticate Modal CLI**: Connect your local environment to your Modal account.
   ```bash
   python -m modal token new
   ```
5. **Hugging Face Token**: You need a Hugging Face token with Read permissions to download the model.

---

## 🔐 Step 1: Configure Secrets in Modal

Your application requires access to Hugging Face to download the model. Instead of hardcoding tokens, Modal uses Secrets.

Create a secret named `huggingface-secret` in your Modal workspace:

```bash
python -m modal secret create huggingface-secret HF_TOKEN=your_hugging_face_token_here
```

*Note: This only needs to be done once per Modal workspace.*

---

## 📦 Step 2: Pre-cache the Model Weights

To avoid paying for the model download time on every cold start, we use a Modal Persistent Volume (`genz-translator-cache`). 

Run the caching script once to download the weights directly into the Modal Volume:

```bash
python -m modal run modal_app.py::download_model
```

You should see logs indicating that the model is downloading. Once it says `✅ Model downloaded and cached in Modal Volume`, you're ready to deploy.

---

## 🚀 Step 3: Development & Serving

If you want to test the API endpoint live while making code changes locally, use `modal serve`. This command sets up an ephemeral endpoint and live-reloads whenever you save changes to your code.

```bash
python -m modal serve modal_app.py
```

* **Live URL**: Look for the `fastapi_app => https://...` URL in your terminal.
* **Stop**: Press `Ctrl+C` to tear down the ephemeral deployment.

---

## 🌍 Step 4: Production Deployment

When you are ready to make the API permanent, use the `deploy` command:

```bash
python -m modal deploy modal_app.py
```

This will output a permanent `modal.run` URL that you can use in your frontend applications. It will look something like this:
`https://your-workspace--genz-translator-fastapi-app.modal.run`

---

## 🧪 Step 5: Testing the Deployed API

A test script is provided to verify your deployed endpoints.

1. Open `test_modal_api.py`.
2. Ensure the `BASE_URL` is set to your newly deployed Modal endpoint.
3. Run the test script:

```bash
python test_modal_api.py
```

This will automatically test the `/health` and `/translate` endpoints and print the outputs.

### API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns the model loading status, VRAM usage, and GPU info. |
| `POST` | `/translate` | Translates text. Body: `{"text": "...", "max_new_tokens": 100, "temperature": 0.7}` |
| `GET` | `/docs` | Interactive Swagger UI for testing the API in your browser. |

---

## 💡 Troubleshooting

* **ModuleNotFoundError: No module named 'app'**  
  If you encounter this during deployment, it usually means your image cache is stale. In `modal_app.py`, change the value of `FORCE_REBUILD_CACHE` in the `.env()` chain to force Modal to rebuild the container image.
  
* **Deployment hangs or fails instantly**  
  Check your Modal dashboard logs. Ensure your `HF_TOKEN` secret was created properly and has read access.

* **High latency on the first request**  
  This is called a "cold start." It takes ~15-30 seconds to boot the container and load the model into VRAM. After the first request, the container stays "warm" and subsequent requests take 1-3 seconds. The app is currently configured with `min_containers=1` to keep one instance permanently warm (costing ~$0.15/hr), but you can remove this in `modal_app.py` if you prefer to scale to zero and save money.
