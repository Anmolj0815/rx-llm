FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements.txt .

# Ensure old pinecone-client is removed, then install all dependencies
RUN pip uninstall -y pinecone-client || true && \
    pip install --no-cache-dir -r requirements.txt

# --- NEW STEP: Pre-download the embedding model ---
# Set the HuggingFace cache directory to a known writable location within the app
ENV HF_HOME /app/.hf_cache
# Run a Python command to download the model.
# This ensures the model is cached inside the Docker image during build time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the entire application code into the working directory
COPY . .

# Explicitly set the PYTHONPATH to include the current working directory.
ENV PYTHONPATH /app

# Expose the port FastAPI listens on. Render will detect this.
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
