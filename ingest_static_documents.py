# ingest_static_documents.py
import os
from dotenv import load_dotenv
import asyncio
import uuid

# LangChain specific imports for ingestion
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec # Direct Pinecone client for index management


# Custom Utility Imports (if needed, or replicate logic)
from utils.document_parser import download_file # For downloading files

load_dotenv()

# --- Configuration from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") # Needed for HuggingFaceEmbeddings endpoint if used

# Check for essential environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not set in .env. For ServerlessSpec, this should be a region like 'us-east-1'.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")
if not HF_TOKEN: # If using HuggingFaceEndpointEmbeddings
    print("Warning: HF_TOKEN not set. If using HuggingFaceEndpointEmbeddings, it might fail. Falling back to local model if possible.")


# --- IMPORTANT: Your actual public URLs for static policy documents ---
STATIC_DOCUMENT_URLS = [
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    # Add your other static policy document URLs here:
    # "https://your-storage-account.blob.core.windows.net/your-container/another_policy.pdf",
]

STATIC_DOCS_NAMESPACE = "static-policies"

# Initialize Embeddings (must match main.py's choice and dimension)
embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/tmp/.cache"
    )
    print("✅ Local SentenceTransformer Embeddings initialized for ingestion.")
except Exception as e:
    print(f"❌ Error initializing local embeddings for ingestion: {e}")
    # Fallback to endpoint embeddings if local fails, requires HF_TOKEN
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            huggingfacehub_api_token=HF_TOKEN
        )
        print("✅ Fallback HuggingFace Endpoint Embeddings initialized for ingestion.")
    except Exception as e2:
        print(f"❌ All embedding methods failed for ingestion: {e2}. Cannot ingest documents.")
        exit(1) # Critical failure for ingestion script

# LangChain's text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""], # Simpler separators for base ingestion
    length_function=len,
    keep_separator=True
)

async def ingest_static_documents_to_pinecone():
    print("Starting static document ingestion to Pinecone...")
    pinecone_client_instance = None
    langchain_pinecone_index = None

    try:
        # Initialize direct Pinecone client for index management
        pinecone_client_instance = Pinecone(api_key=PINECONE_API_KEY)
        print("Pinecone client instance created for ingestion.")

        # Check/create index as per your previous helper
        existing_indexes_names = pinecone_client_instance.list_indexes().names()
        if PINECONE_INDEX_NAME not in existing_indexes_names:
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}...")
            pinecone_client_instance.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=embeddings.client.get_sentence_embedding_dimension(), # Get dimension from initialized model
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")
        else:
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

        # Initialize LangChain Pinecone vector store for data operations
        langchain_pinecone_index = LangChainPinecone(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        print("✅ LangChain Pinecone vector store initialized for ingestion.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Pinecone for ingestion: {e}")
        return

    # Process each static document
    for doc_url in STATIC_DOCUMENT_URLS:
        document_id = str(uuid.uuid4()) # Unique ID for each static document
        try:
            print(f"Processing static document for ingestion: {doc_url}")
            
            # Download and extract content
            file_stream = await asyncio.to_thread(download_file, doc_url)
            # Write to temp file for PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_stream.read())
                temp_path = temp_file.name
            
            loader = PyPDFLoader(temp_path)
            # Load documents as LangChain Documents
            # Using asyncio.to_thread as loader.load() is synchronous
            lc_docs = await asyncio.to_thread(loader.load)
            
            # Add metadata for traceability (e.g., original URL, unique ID)
            for doc in lc_docs:
                doc.metadata['source'] = doc_url
                doc.metadata['document_id'] = document_id
                
            # Chunk the documents
            chunks = await asyncio.to_thread(text_splitter.split_documents, lc_docs)
            
            if not chunks:
                print(f"Skipping {doc_url}: No chunks generated.")
                continue

            # Upsert to Pinecone with the specific namespace
            # LangChain Pinecone `add_documents` is synchronous, run in threadpool
            await asyncio.to_thread(
                langchain_pinecone_index.add_documents,
                chunks,
                namespace=STATIC_DOCS_NAMESPACE # Target static namespace
            )
            os.unlink(temp_path) # Clean up temp file
            print(f"Successfully ingested {doc_url} ({len(chunks)} chunks) into namespace '{STATIC_DOCS_NAMESPACE}'.")
        except Exception as e:
            print(f"Failed to ingest {doc_url}: {e}")
            # Continue with other documents even if one fails

    print("Static document ingestion complete.")

if __name__ == "__main__":
    print("Starting static document ingestion process for base knowledge...")
    asyncio.run(ingest_static_documents_to_pinecone())
