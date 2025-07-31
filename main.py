import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
import time
import json
import re
from urllib.parse import urlparse
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil 

# --- LangChain specific imports ---
# CRITICAL: Ensure this is the correct import for the langchain-pinecone package
from langchain_pinecone import Pinecone as LangChainPineconeVectorStore # <--- CRITICAL CHANGE: Correct import
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec # Direct Pinecone client for index management


# --- Custom Utility Imports ---
from utils.document_parser import download_file, chunk_text

# Load environment variables from .env file
load_dotenv()

# Suppress tokenizer parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'false')

# --- Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN") # For HuggingFace embeddings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
AUTH_TOKEN = os.getenv("AUTH_TOKEN") 

# Check for essential environment variables
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not set. Endpoint embeddings might fail if used.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY environment variable not set.")
if not PINECONE_ENVIRONMENT:
    raise RuntimeError("PINECONE_ENVIRONMENT environment variable not set (e.g., 'us-east-1').")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME environment variable not set for Pinecone index.")
if not AUTH_TOKEN:
    raise RuntimeError("AUTH_TOKEN environment variable not set. Please set a secret token for authentication.")

# --- Shared Global Components ---
embeddings = None
llm = None
pinecone_client_instance = None 
langchain_pinecone_index = None # This will be the LangChainPineconeVectorStore instance
global_static_bm25_retriever = None 
global_static_documents_for_bm25 = []

# Thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# --- FastAPI App Setup ---
app = FastAPI(
    title="HackRx 6.0 Insurance RAG Backend",
    version="2.0.0",
    description="Advanced RAG Backend with Insurance Claim Decision Engine and Structured Analysis using Pinecone."
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

security = HTTPBearer()

# --- Async Helper for ThreadPoolExecutor ---
async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, func, *args, **kwargs)

# --- Startup Event: Initialize LLM, Embeddings, Pinecone ---
@app.on_event("startup")
async def startup_event():
    global embeddings, llm, pinecone_client_instance, langchain_pinecone_index, global_static_bm25_retriever, global_static_documents_for_bm25
    print("🚀 Initializing core components...")

    # 1. Initialize Embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="/app/.hf_cache" # CRITICAL: Point to the pre-downloaded location
        )
        print("✅ Local SentenceTransformer Embeddings initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing local embeddings: {e}")
        try:
            embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-mpnet-base-v2", 
                huggingfacehub_api_token=HF_TOKEN 
            )
            print("✅ Fallback HuggingFace Endpoint Embeddings initialized successfully.")
        except Exception as e2:
            print(f"❌ All embedding methods failed: {e2}. System will not function.")
            embeddings = None 

    # 2. Initialize LLM (Groq)
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=GROQ_API_KEY,
            temperature=0
        )
        print("✅ LLM (Groq) initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}. System will not function.")
        llm = None 

    # 3. Initialize Pinecone Client and Index
    if embeddings is not None: # Only proceed if embeddings are ready
        try:
            pinecone_client_instance = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone client instance created.")

            existing_indexes_names = pinecone_client_instance.list_indexes().names()
            if PINECONE_INDEX_NAME not in existing_indexes_names:
                print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}...")
                pinecone_client_instance.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=embeddings.client.get_sentence_embedding_dimension(), 
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
                )
                print(f"New Pinecone index '{PINECONE_INDEX_NAME}' created.")
            else:
                print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

            # CRITICAL CHANGE: Initialize LangChain Pinecone vector store using the pre-initialized client
            # This avoids arguments issues with Pinecone.__init__() by passing the client object directly.
            langchain_pinecone_index = LangChainPineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY, # Pass it explicitly
                environment=PINECONE_ENVIRONMENT,  # Pass environment explicitly
                # Pass the direct client instance if it's available and compatible
                _text_key="text" # Standard metadata key for text in Pinecone from LangChain
            )
            # A more robust way to use an existing client might be:
            # langchain_pinecone_index = LangChainPineconeVectorStore.from_existing_index(
            #     index_name=PINECONE_INDEX_NAME,
            #     embedding=embeddings,
            #     pinecone_api_key=PINECONE_API_KEY,
            #     environment=PINECONE_ENVIRONMENT # Ensure environment is also passed here
            # )
            # However, the direct constructor is usually fine if args are correct.
            # The issue seems to be the version interaction.
            # Let's try passing the client instance directly IF Langchain Pinecone supports it.
            # It generally uses `api_key` and `environment` directly in its constructor.

            # Reverting to the simpler, standard constructor as per docs
            langchain_pinecone_index = LangChainPineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT # <--- Ensure this is passed
            )
            print("✅ LangChain Pinecone vector store initialized.")

            print(f"Attempting to load documents from '{STATIC_DOCS_NAMESPACE}' for BM25...")
            try:
                global_static_documents_for_bm25 = [] 
                global_static_bm25_retriever = BM25Retriever.from_documents(global_static_documents_for_bm25)
                global_static_bm25_retriever.k = 3 
                print("✅ BM25 retriever initialized for static content (will update dynamically).")
            except Exception as e:
                print(f"⚠️ Could not initialize BM25 for static documents: {e}. Will proceed without it initially for static.")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Pinecone index or client on startup: {e}. "
                  "Vector search will be unavailable.")
            langchain_pinecone_index = None 
    else:
        print("Pinecone initialization skipped due to embedding model failure.")

    print("✅ All primary components initialization attempt complete.")

# --- Pydantic Models for Request/Response (No Change) ---
class DebugRequest(BaseModel):
    question: str

class ClaimRequest(BaseModel):
    documents: Union[List[str], str] 
    claim_details: Dict[str, Any] = Field(default_factory=dict)
    questions: List[str]

class CoordinationOfBenefits(BaseModel):
    has_other_insurance: bool = False
    primary_insurance: Optional[str] = None
    secondary_insurance: Optional[str] = None
    primary_payment: Optional[float] = None
    remaining_amount: Optional[float] = None

class ClaimDecision(BaseModel):
    question: str
    decision: str  
    confidence_score: float = Field(ge=0.0, le=1.0)
    payout_amount: Optional[float] = None
    reasoning: str
    policy_sections_referenced: List[str] = Field(default_factory=list)
    exclusions_applied: List[str] = Field(default_factory=list)
    coordination_of_benefits: Optional[CoordinationOfBenefits] = None
    processing_notes: List[str] = Field(default_factory=list)

class ProcessingMetadata(BaseModel):
    request_id: str
    processing_time: float
    chunks_analyzed: int
    model_used: str
    timestamp: str

class EnhancedAnswerResponse(BaseModel):
    decisions: List[ClaimDecision]
    processing_metadata: ProcessingMetadata
    audit_trail: List[str] = Field(default_factory=list)

# Enhanced Insurance-Specific Prompt Template (No Change)
INSURANCE_CLAIM_PROMPT = """
You are an expert insurance claim processor with deep knowledge of policy terms, coverage rules, and claim evaluation. You must analyze claims systematically and provide structured decisions.<br>

ANALYSIS FRAMEWORK:<br>
1. **Eligibility Assessment**: Determine if the claim is covered under the policy<br>
2. **Coverage Limits**: Identify applicable limits, deductibles, and caps<br>
3. **Coordination of Benefits**: Check for multiple insurance policies and calculate remaining amounts<br>
4. **Exclusion Review**: Identify any policy exclusions that apply<br>
5. **Decision Logic**: Apply business rules to determine approval/denial<br>
6. **Payout Calculation**: Calculate exact amounts considering all factors<br>

RESPONSE FORMAT (Must be valid JSON):<br>
{{<br>
    "decision": "[APPROVED/DENIED/PENDING_REVIEW]",<br>
    "confidence_score": [0.0-1.0],<br>
    "payout_amount": [amount or null],<br>
    "reasoning": "Detailed explanation with specific policy references",<br>
    "policy_sections_referenced": ["section1", "section2"],
    "exclusions_applied": ["exclusion1", "exclusion2"],
    "coordination_of_benefits": {{<br>
        "has_other_insurance": [true/false],<br>
        "primary_insurance": "name or null",<br>
        "secondary_insurance": "name or null", <br>
        "primary_payment": [amount or null],<br>
        "remaining_amount": [amount or null]<br>
    }},<br>
    "processing_notes": ["note1", "note2"]<br>
}}<br>

IMPORTANT RULES:<br>
- Base decisions ONLY on information in the policy context<br>
- For coordination of benefits, calculate remaining amounts after primary insurance<br>
- Include confidence scores based on clarity of policy language<br>
- Reference specific policy sections in your reasoning<br>
- If information is unclear, use "PENDING_REVIEW" decision<br>
- Strictly adhere to the JSON format. Do not add any text before or after the JSON.<br>

Policy Context:<br>
{context}<br>

Claim Question: {question}<br>

Insurance Analysis (JSON format only):<br>
"""

ENHANCED_PROMPT = PromptTemplate(
    template=INSURANCE_CLAIM_PROMPT,
    input_variables=["context", "question"]
)

class InsuranceDecisionEngine:
    """Core decision engine for insurance claim processing (No Change)"""
    
    def __init__(self):
        self.decision_rules = {
            'min_confidence_for_approval': 0.7,
            'max_payout_without_review': 10000,
            'coordination_keywords': [
                'coordination of benefits', 'other insurance', 'secondary claim',
                'primary insurance', 'remaining amount', 'balance claim'
            ],
            'exclusion_keywords': [
                'excluded', 'not covered', 'limitation', 'restriction'
            ]
        }
    
    def extract_financial_amounts(self, text: str) -> List[float]:
        amounts = re.findall(r'\$?[\d,]+\.?\d*', text)
        return [float(amt.replace('$', '').replace(',', '')) for amt in amounts if amt]
    
    def detect_coordination_of_benefits(self, context: str, question: str) -> bool:
        combined_text = (context + " " + question).lower()
        return any(keyword in combined_text for keyword in self.decision_rules['coordination_keywords'])
    
    def calculate_confidence_score(self, context: str, decision_factors: Dict) -> float:
        score = 0.5
        
        if decision_factors.get('policy_sections_referenced'):
            score += 0.2
        
        if decision_factors.get('has_coordination'):
            score -= 0.1
        
        if decision_factors.get('has_amounts'):
            score += 0.1
        
        return max(0.0, min(1.0, score))

class HybridRetriever:
    """Enhanced retrieval system combining semantic and keyword search (No Change)"""
    
    def __init__(self, vector_store, documents_for_bm25: List[Document]):
        self.vector_store = vector_store
        self.documents_for_bm25 = documents_for_bm25
        self.bm25_retriever = None
        self.setup_bm25_retriever()
    
    def setup_bm25_retriever(self):
        try:
            doc_texts = [doc.page_content for doc in self.documents_for_bm25]
            if doc_texts:
                self.bm25_retriever = BM25Retriever.from_texts(doc_texts)
                self.bm25_retriever.k = 3
                print("✅ BM25 retriever initialized/updated.")
            else:
                print("⚠️ No documents for BM25 initialization/update. BM25 will not be used.")
                self.bm25_retriever = None
        except Exception as e:
            print(f"⚠️ BM25 retriever failed to initialize/update, using vector-only: {e}")
            self.bm25_retriever = None
    
    async def retrieve_relevant_docs(self, query: str, k: int = 6) -> List[Document]:
        all_docs = []
        
        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "fetch_k": k * 2}
        )
        vector_docs = await run_in_threadpool(vector_retriever.get_relevant_documents, query)
        all_docs.extend(vector_docs)
        
        if self.bm25_retriever:
            try:
                bm25_docs = await run_in_threadpool(self.bm25_retriever.get_relevant_documents, query)
                all_docs.extend(bm25_docs)
            except Exception as e:
                print(f"⚠️ BM25 retrieval failed: {e}")
        
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:k]

# Initialize decision engine
decision_engine = InsuranceDecisionEngine()

# Helper functions (No Change, except adding tempfile/shutil imports at top)
def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

async def extract_document_content_from_url(url: str) -> str:
    """Downloads and extracts text content from a URL (currently only PDF)."""
    try:
        print(f"📥 Downloading document from: {url}")
        file_stream = await run_in_threadpool(download_file, url)

        if url.lower().endswith('.pdf') or 'pdf' in url.lower():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_stream.read())
                temp_path = temp_file.name
            
            loader = PyPDFLoader(temp_path)
            pages = await run_in_threadpool(loader.load)
            
            content = ""
            for i, page in enumerate(pages):
                page_text = page.page_content.strip()
                if page_text:
                    content += f"\n--- Page {i+1} ---\n{page_text}\n"
            
            os.unlink(temp_path)
            
            content = content.replace('\n\n\n', '\n\n')
            content = content.replace('\t', ' ')
            
            print(f"✅ PDF extracted successfully. Content length: {len(content)} characters")
            return content
        else:
            print(f"Warning: Attempting generic text extraction for unsupported URL type: {url}")
            return file_stream.getvalue().decode('utf-8', errors='ignore').strip()
            
    except Exception as e:
        print(f"❌ Error extracting document content from {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract document content from {url}: {str(e)}")

async def process_incoming_documents(documents: Union[List[str], str]) -> List[Document]:
    """Processes incoming document URLs or raw text and returns a list of LangChain Documents."""
    if isinstance(documents, str):
        documents = [documents]
    
    all_processed_content = []
    
    for doc_source in documents:
        if content.strip():
            all_processed_content.append(content)
        else:
            print(f"Warning: No meaningful content extracted from '{doc_source}'.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n### ",
            "\n\nSection ",
            "\n\nClause ",
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
        length_function=len,
        keep_separator=True
    )
    
    if not all_processed_content:
        return []

    combined_raw_text = "\n\n".join(all_processed_content)
    base_doc = Document(page_content=combined_raw_text)
    
    chunks = await run_in_threadpool(text_splitter.split_documents, [base_doc])
    print(f"Created {len(chunks)} chunks from incoming documents.")
    return chunks

def parse_llm_response(response_text: str) -> Dict:
    """Parse structured JSON response from LLM, with robust fallback. (No Change)"""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print(f"⚠️ LLM response did not contain valid JSON. Raw: {response_text[:200]}...")
            return {
                "decision": "PENDING_REVIEW",
                "confidence_score": 0.5,
                "payout_amount": None,
                "reasoning": response_text,
                "policy_sections_referenced": [],
                "exclusions_applied": [],
                "coordination_of_benefits": {
                    "has_other_insurance": False,
                    "primary_insurance": None,
                    "secondary_insurance": None,
                    "primary_payment": None,
                    "remaining_amount": None
                },
                "processing_notes": ["LLM response parsing required fallback method. Expected JSON."]
            }
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}. Raw response: {response_text[:200]}...")
        return {
            "decision": "PENDING_REVIEW",
            "confidence_score": 0.3,
            "payout_amount": None,
            "reasoning": f"Error parsing response: {response_text}",
            "policy_sections_referenced": [],
            "exclusions_applied": [],
            "coordination_of_benefits": {
                "has_other_insurance": False,
                "primary_insurance": None,
                "secondary_insurance": None,
                "primary_payment": None,
                "remaining_amount": None
            },
            "processing_notes": [f"JSON parsing error: {str(e)}"]
        }

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    received_token = credentials.credentials
    expected_token = AUTH_TOKEN
    
    if not expected_token:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: AUTH_TOKEN not set.")

    if received_token == expected_token or received_token.strip() == expected_token.strip():
        return received_token
    
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired token.")


# --- API Endpoints (No Change in logic, only updated Pinecone calls) ---
@app.get("/")
def root():
    return {
        "message": "HackRx 6.0 Insurance RAG Backend with Decision Engine (Pinecone Integrated)",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Insurance claim decision engine",
            "Coordination of benefits analysis",
            "Structured JSON responses",
            "Hybrid retrieval (Vector - Pinecone + BM25)",
            "Confidence scoring",
            "Audit trail support",
            "Persistent Vector Store (Pinecone)"
        ],
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "rag_status": "/rag-status",
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "decision_engine_status": "/decision-engine-status"
        }
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint to confirm the service is running and core components are ready."""
    status_info = {
        "status": "healthy",
        "embeddings_ready": embeddings is not None,
        "llm_ready": llm is not None,
        "pinecone_client_ready": pinecone_client_instance is not None,
        "langchain_pinecone_index_ready": langchain_pinecone_index is not None,
        "decision_engine_ready": decision_engine is not None,
        "static_bm25_retriever_ready": global_static_bm25_retriever is not None
    }
    
    if langchain_pinecone_index:
        try:
            stats = await run_in_threadpool(pinecone_client_instance.describe_index_stats, index_name=PINECONE_INDEX_NAME)
            status_info["pinecone_stats"] = stats.to_dict()
            status_info["message"] = "Service, LLM, Embeddings, and Pinecone connected."
        except Exception as e:
            status_info["status"] = "error"
            status_info["message"] = f"Pinecone connectivity issue: {e}"
            status_info["langchain_pinecone_index_ready"] = False
    else:
        status_info["message"] = "Service running, but Pinecone index not fully initialized."
        status_info["status"] = "warning" if embeddings and llm else "error"

    return status_info

@app.get("/decision-engine-status")
def decision_engine_status():
    """Check decision engine configuration and rules (No Change)"""
    return {
        "engine_active": True,
        "decision_rules": decision_engine.decision_rules,
        "supported_decisions": ["APPROVED", "DENIED", "PENDING_REVIEW"],
        "coordination_benefits_supported": True,
        "confidence_scoring_enabled": True
    }

@app.post("/debug-search")
async def debug_search(request: DebugRequest):
    """Enhanced debug endpoint with hybrid retrieval information (No Change in logic)"""
    if langchain_pinecone_index is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pinecone vector store not initialized.")
    if embeddings is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding model not initialized.")

    try:
        temp_hybrid_retriever = HybridRetriever(langchain_pinecone_index, global_static_documents_for_bm25)
        docs = await temp_hybrid_retriever.retrieve_relevant_docs(request.question, k=6)
        
        retrieved_chunks = []
        for i, doc in enumerate(docs):
            retrieved_chunks.append({
                "chunk_id": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "full_length": len(doc.page_content),
                "source": doc.metadata.get('source', 'unknown')
            })
        
        has_cob = decision_engine.detect_coordination_of_benefits(
            " ".join([doc.page_content for doc in docs]), 
            request.question
        )
        
        return {
            "question": request.question,
            "total_chunks_retrieved": len(docs),
            "chunks": retrieved_chunks,
            "decision_engine_analysis": {
                "coordination_of_benefits_detected": has_cob,
                "retrieval_method": "hybrid" if temp_hybrid_retriever.bm25_retriever else "vector_only"
            }
        }
        
    except Exception as e:
        print(f"Error during debug search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Debug search error: {str(e)}")

@app.post("/hackrx/run", response_model=EnhancedAnswerResponse)
async def run_enhanced_query(request: ClaimRequest, token: str = Depends(verify_token)):
    global langchain_pinecone_index, embeddings, llm, global_static_bm25_retriever, global_static_documents_for_bm25
    
    if embeddings is None or llm is None or langchain_pinecone_index is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Core components (Embeddings, LLM, or Pinecone) not initialized. Service is unavailable.")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    audit_trail = [f"Request {request_id} started at {datetime.now().isoformat()}"]
    
    request_specific_docs: List[Document] = []
    dynamic_doc_namespace = f"request-{request_id}"

    try:
        # Step 1: Process incoming documents (from the request body)
        if request.documents:
            print(f"Processing {len(request.documents) if isinstance(request.documents, list) else 1} incoming document(s).")
            request_specific_docs = await process_incoming_documents(request.documents)
            audit_trail.append(f"Processed {len(request_specific_docs)} LangChain Documents from request.documents.")

            if request_specific_docs:
                await run_in_threadpool(
                    langchain_pinecone_index.add_documents,
                    request_specific_docs,
                    namespace=dynamic_doc_namespace
                )
                print(f"Added {len(request_specific_docs)} chunks to Pinecone under namespace '{dynamic_doc_namespace}'.")
                audit_trail.append(f"Ingested {len(request_specific_docs)} chunks to Pinecone (dynamic namespace).")
            else:
                audit_trail.append("No valid LangChain Documents generated from request.documents.")
        else:
            audit_trail.append("No 'documents' provided in the request for dynamic ingestion.")

        # Step 2: Process each question with enhanced decision logic
        decisions = []
        
        for question in request.questions:
            try:
                print(f"Processing question: {question}")
                audit_trail.append(f"Processing question: {question[:50]}...")
                
                # First, query the STATIC_DOCS_NAMESPACE
                results_static_pinecone = await run_in_threadpool(
                    langchain_pinecone_index.index.query,
                    vector=embeddings.embed_query(question),
                    top_k=5,
                    include_metadata=True,
                    namespace=STATIC_DOCS_NAMESPACE
                )
                static_relevant_docs = [
                    Document(page_content=match.metadata['text'], metadata={'source': match.metadata.get('source', 'static_pinecone'), 'document_id': match.metadata.get('document_id', 'static_doc')})
                    for match in results_static_pinecone.matches if 'text' in match.metadata
                ]
                
                # Second, query the DYNAMIC_DOC_NAMESPACE (if documents were ingested for this request)
                dynamic_relevant_docs = []
                if request_specific_docs:
                    results_dynamic_pinecone = await run_in_threadpool(
                        langchain_pinecone_index.index.query,
                        vector=embeddings.embed_query(question),
                        top_k=3,
                        include_metadata=True,
                        namespace=dynamic_doc_namespace
                    )
                    dynamic_relevant_docs = [
                        Document(page_content=match.metadata['text'], metadata={'source': match.metadata.get('source', 'dynamic_pinecone'), 'document_id': match.metadata.get('document_id', 'dynamic_doc')})
                        for match in results_dynamic_pinecone.matches if 'text' in match.metadata
                    ]
                
                # Third, create a BM25 retriever only for the documents processed in this specific request
                bm25_docs = []
                if request_specific_docs:
                    try:
                        local_bm25_retriever = await run_in_threadpool(BM25Retriever.from_documents, request_specific_docs)
                        local_bm25_retriever.k = 2
                        bm25_docs = await run_in_threadpool(local_bm25_retriever.get_relevant_documents, question)
                    except Exception as e:
                        print(f"⚠️ Local BM25 retrieval for dynamic docs failed: {e}")

                # Combine all retrieved documents
                all_relevant_docs = static_relevant_docs + dynamic_relevant_docs + bm25_docs
                unique_docs = []
                seen_content = set()
                for doc in all_relevant_docs:
                    if doc.page_content not in seen_content:
                        unique_docs.append(doc)
                        seen_content.add(doc.page_content)
                
                context_docs = unique_docs[:8]
                context = "\n\n".join([doc.page_content for doc in context_docs])
                
                if not context.strip():
                    decisions.append(ClaimDecision(
                        question=question,
                        decision="PENDING_REVIEW",
                        confidence_score=0.1,
                        payout_amount=None,
                        reasoning="No relevant policy information found for this question from any source.",
                        processing_notes=["No relevant documents retrieved from Pinecone or BM25."]
                    ))
                    audit_trail.append("No relevant documents found for LLM context.")
                    continue

                formatted_prompt = ENHANCED_PROMPT.format(context=context, question=question)
                
                llm_result = await run_in_threadpool(llm.invoke, formatted_prompt)
                response_text = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)
                
                parsed_response = parse_llm_response(response_text)
                
                decision = ClaimDecision(
                    question=question,
                    decision=parsed_response.get("decision", "PENDING_REVIEW"),
                    confidence_score=parsed_response.get("confidence_score", 0.5),
                    payout_amount=parsed_response.get("payout_amount"),
                    reasoning=parsed_response.get("reasoning", "Analysis completed"),
                    policy_sections_referenced=parsed_response.get("policy_sections_referenced", []),
                    exclusions_applied=parsed_response.get("exclusions_applied", []),
                    processing_notes=parsed_response.get("processing_notes", [])
                )
                
                cob_data = parsed_response.get("coordination_of_benefits", {})
                if cob_data and cob_data.get("has_other_insurance"):
                    decision.coordination_of_benefits = CoordinationOfBenefits(**cob_data)
                
                decisions.append(decision)
                audit_trail.append(f"Decision generated: {decision.decision} (confidence: {decision.confidence_score})")
                
            except Exception as e:
                print(f"❌ Error processing question '{question}': {str(e)}")
                decisions.append(ClaimDecision(
                    question=question,
                    decision="PENDING_REVIEW",
                    confidence_score=0.0,
                    payout_amount=None,
                    reasoning=f"Error during processing: {str(e)}",
                    processing_notes=[f"Processing error: {str(e)}"]
                ))
                audit_trail.append(f"Error processing question: {str(e)}")
        
        processing_time = time.time() - start_time
        audit_trail.append(f"Processing completed in {processing_time:.2f} seconds")
        
        metadata = ProcessingMetadata(
            request_id=request_id,
            processing_time=processing_time,
            chunks_analyzed=len(request_specific_docs),
            model_used=llm.model_name,
            timestamp=datetime.now().isoformat()
        )
        
        return EnhancedAnswerResponse(
            decisions=decisions,
            processing_metadata=metadata,
            audit_trail=audit_trail
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ Fatal error in enhanced RAG processing: {str(e)}")
        audit_trail.append(f"Fatal error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Fatal RAG Error: {str(e)}")
    finally:
        # CRITICAL FIX: Correct way to delete a namespace using pinecone-client v3.2.2
        try:
            if pinecone_client_instance and dynamic_doc_namespace:
                print(f"Attempting to delete dynamic document namespace: {dynamic_doc_namespace}")
                # Get the index object first, then call delete on it
                index_to_delete_namespace = pinecone_client_instance.Index(PINECONE_INDEX_NAME)
                await run_in_threadpool(
                    index_to_delete_namespace.delete,
                    delete_all=True, # Important: to delete all vectors in the namespace
                    namespace=dynamic_doc_namespace
                )
                print(f"Cleaned up dynamic document namespace: {dynamic_doc_namespace}")
        except Exception as e:
            print(f"Error during dynamic document cleanup for {dynamic_doc_namespace}: {e}")

# --- Health Check (No Change in logic) ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint to confirm the service is running and core components are ready."""
    status_info = {
        "status": "healthy",
        "embeddings_ready": embeddings is not None,
        "llm_ready": llm is not None,
        "pinecone_client_ready": pinecone_client_instance is not None,
        "langchain_pinecone_index_ready": langchain_pinecone_index is not None,
        "decision_engine_ready": decision_engine is not None,
        "static_bm25_retriever_ready": global_static_bm25_retriever is not None
    }
    
    if langchain_pinecone_index:
        try:
            stats = await run_in_threadpool(pinecone_client_instance.describe_index_stats, index_name=PINECONE_INDEX_NAME)
            status_info["pinecone_stats"] = stats.to_dict()
            status_info["message"] = "Service, LLM, Embeddings, and Pinecone connected."
        except Exception as e:
            status_info["status"] = "error"
            status_info["message"] = f"Pinecone connectivity issue: {e}"
            status_info["langchain_pinecone_index_ready"] = False
    else:
        status_info["message"] = "Service running, but Pinecone index not fully initialized."
        status_info["status"] = "warning" if embeddings and llm else "error"

    return status_info

if __name__ == "__main__":
    import uvicorn
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting Combined HackRx 6.0 RAG Backend Server...")
    print(f"📍 Server will be available at: http://{HOST}:{PORT}")
    print("🎯 HackRx 6.0 Features:")
    print("   - Insurance claim decision engine")
    print("   - Coordination of benefits analysis")
    print("   - Structured JSON responses")
    print("   - Hybrid retrieval (Vector - Pinecone + BM25)")
    print("   - Audit trail and processing metadata")
    print("   - Policy section referencing")
    print("   - Persistent Vector Store (Pinecone) for base knowledge")
    
    uvicorn.run(app, host=HOST, port=PORT)
