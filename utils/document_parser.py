# utils/document_parser.py
import io
import requests
# PyPDFLoader from LangChain will handle PDF extraction, so direct pypdf import here is not strictly needed for the main flow.
# from pypdf import PdfReader # Not directly used if PyPDFLoader is used in main.py

def download_file(url: str) -> io.BytesIO:
    """Downloads a file from a given URL and returns it as a BytesIO object."""
    try:
        response = requests.get(url, timeout=30) # Increased timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return io.BytesIO(response.content)
    except requests.exceptions.Timeout:
        print(f"Error: Download timed out for {url}")
        raise ValueError(f"File download timed out for {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        raise ValueError(f"Failed to download file from {url}: {e}")

# Note: extract_text_from_pdf and parse_document logic for specific file types
# will be consolidated within main.py using LangChain's loaders for better integration.
# This file now primarily provides core utilities like download_file and chunk_text.

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks. Handles empty or short texts gracefully."""
    if not text or len(text.strip()) == 0:
        return []

    text_len = len(text)
    if text_len <= chunk_size:
        return [text]

    chunks = []
    current_start = 0
    while current_start < text_len:
        chunk_end = min(current_start + chunk_size, text_len)
        chunk = text[current_start:chunk_end]
        chunks.append(chunk)

        if chunk_end == text_len:
            break # Reached end of text

        current_start += (chunk_size - chunk_overlap)
        if current_start >= text_len:
            break

    return chunks
