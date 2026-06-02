from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv(override=True)
token = os.getenv("MODEL_WRITER_TOKEN")
api = HfApi(token=token)
repo_id = "Bhargavreddy1/Llama-3.2-1B-GenZ-Translator-v1"

print(f"Uploading merged model from './tests/llama3_genz_final' to '{repo_id}'...")
api.upload_folder(
    folder_path="./tests/llama3_genz_final",
    repo_id=repo_id,
    repo_type="model",
)
print("Upload complete!")
