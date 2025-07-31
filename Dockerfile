FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements.txt .

# Ensure old pinecone-client is removed, then install all dependencies
# This '|| true' makes sure the build doesn't fail if pinecone-client isn't found.
RUN pip uninstall -y pinecone-client || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the working directory
COPY . .

# Explicitly set the PYTHONPATH to include the current working directory.
ENV PYTHONPATH /app

# Expose the port FastAPI listens on. Render will detect this.
# This must match the port Uvicorn listens on in the CMD.
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn.
# Uvicorn will listen on port 8000 inside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
