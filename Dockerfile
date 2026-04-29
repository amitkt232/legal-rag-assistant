# Why Python 3.11-slim?
# Slim = smaller image size (~150MB vs ~900MB for full Python)
# 3.11 = stable, well-supported, compatible with all our packages
# Smaller image = faster deployment, less attack surface
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Why install system dependencies first?
# Docker caches each layer. System deps change rarely.
# By installing them first, this layer is cached and
# not rebuilt when only our Python code changes.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — another caching optimisation
# If requirements.txt has not changed, pip install
# layer is cached. Only rebuilt when deps change.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories needed at runtime
RUN mkdir -p chroma_db data/contracts

# Expose FastAPI port
EXPOSE 8000

# Why use uvicorn directly instead of python main.py?
# In production, we want explicit control over workers.
# --host 0.0.0.0 allows external connections (required in container)
# --port 8000 matches EXPOSE above
# reload=False in production — no file watching needed
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]