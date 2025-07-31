FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements.txt .

# Ensure old pinecone-client is removed, then install all dependencies
RUN pip uninstall -y pinecone-client || true && \
    pip install --no-cache-dir -r requirements.txt

# --- CRITICAL: Final attempt for robust model pre-download and IndentationError fix ---
# Set the HuggingFace cache directory within the app's working directory.
ENV HF_HOME /app/.hf_cache
# Create the cache directory explicitly and ensure it's writable
RUN mkdir -p ${HF_HOME} && chmod -R 777 ${HF_HOME}

# Use python and sentence_transformers to download and cache the model.
# Pay EXTREME attention to the left-alignment here. No spaces before the '\' character.
# Each line of the Python code should start immediately after the backslash, or if it's the first line,
# immediately after the opening double quote of the -c argument.
RUN python -c "import os;\
import sys;\
try:\
    from sentence_transformers import SentenceTransformer;\
    print(f'Attempting to download model to HF_HOME: {os.environ.get(\"HF_HOME\")}');\
    token = os.environ.get('HF_TOKEN');\
    if token:\
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=token);\
    else:\
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2');\
    print('Model downloaded and cached successfully.');\
    import glob;\
    cache_path = os.path.join(os.environ.get('HF_HOME'), 'models');\
    if os.path.exists(cache_path):\
        print(f'Contents of {cache_path}: {os.listdir(cache_path)}');\
        print('Model directory should be present in cache.')\
    else:\
        print('Warning: Model cache directory not found after download attempt.');\
        sys.exit(1);\
except Exception as e:\
    print(f'CRITICAL: Embedding model pre-download FAILED: {e}', file=sys.stderr);\
    sys.exit(1);\
"

# Copy the entire application code into the working directory
COPY . .

# Explicitly set the PYTHONPATH to include the current working directory.
ENV PYTHONPATH /app

# Expose the port FastAPI listens on. Render will detect this.
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
