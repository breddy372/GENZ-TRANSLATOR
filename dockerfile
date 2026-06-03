FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Copy requirements.txt first to leverage docker cache
COPY requirements.txt .

# Install dependencies (will skip torch since it is pre-installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]