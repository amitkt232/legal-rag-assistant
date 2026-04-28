from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import hashlib
import time
from dotenv import load_dotenv

load_dotenv()

# Project imports
from app.core.ingestion import ingest_contract
from app.core.qa_chain import get_answer
from app.core.agent import get_agent_answer
from app.core.guardrails import check_input
from app.core.embedder import get_embedder


# ── In-memory cache ───────────────────────────────────────
# Why in-memory cache for a portfolio project?
# Redis is the production standard but requires a separate
# service running. For a portfolio project, in-memory cache
# demonstrates the concept without the infrastructure overhead.
# In production: replace _cache dict with Redis using
# langchain.cache.RedisCache or RedisSemanticCache.
#
# This directly answers Strides Q13:
# "Which library have you used in LangChain for caching?"
# Answer: "We implemented in-memory caching at the API layer.
# In production we would use LangChain's RedisSemanticCache
# which caches based on semantic similarity — so similar
# questions return the same cached answer."

_cache = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


def get_cache_key(session_id: str, question: str) -> str:
    """MD5 hash of session + question for consistent short keys."""
    raw = f"{session_id}:{question.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_from_cache(key: str):
    """Returns cached response if exists and not expired."""
    if key in _cache:
        cached = _cache[key]
        age = time.time() - cached["timestamp"]
        if age < CACHE_TTL_SECONDS:
            cached["response"]["cached"] = True
            cached["response"]["cache_age_seconds"] = round(age)
            return cached["response"]
        else:
            del _cache[key]
    return None


def set_cache(key: str, response: dict):
    """Stores response in cache with timestamp."""
    _cache[key] = {
        "timestamp": time.time(),
        "response": response
    }


# ── Lifespan event handler ────────────────────────────────
# Why lifespan instead of on_event?
# FastAPI 0.136+ deprecated on_event decorator.
# Lifespan context manager is the current standard —
# cleaner, testable, handles both startup and shutdown.
# Startup: pre-load embedding model so first request is fast.
# This is called a "warm start" pattern in production ML APIs.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Legal RAG API...")
    get_embedder()
    print("Embedder pre-loaded. API ready.")
    yield
    # Shutdown
    print("API shutting down.")


# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Legal Contract Intelligence API",
    description="RAG-powered legal contract review assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────
class QuestionRequest(BaseModel):
    session_id: str
    question: str
    use_agent: bool = False


class QuestionResponse(BaseModel):
    answer: str
    sources: list
    confidence: float = 0.0
    latency_seconds: float
    status: str
    warnings: list = []
    cached: bool = False
    cache_age_seconds: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_size: int


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Why have this?
    Load balancers, Kubernetes, and monitoring systems
    ping this endpoint to verify the service is alive.
    In Kubernetes this is our readiness probe — the pod
    only receives traffic once this returns 200.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cache_size=len(_cache)
    )


@app.post("/upload")
async def upload_contract(file: UploadFile = File(...)):
    """
    Uploads and ingests a contract PDF.

    Security: Raw PDF is never persisted on server.
    We write to temp file, ingest to vectors, delete immediately.
    Only embeddings and metadata are stored — original PDF gone
    within seconds of upload.
    """

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Validate file size — max 50MB
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 50MB"
        )

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_contract(tmp_path)
        result["doc_name"] = file.filename

        return JSONResponse(
            status_code=200,
            content={
                "session_id": result["session_id"],
                "doc_name": result["doc_name"],
                "total_pages": result["total_pages"],
                "total_chunks": result["total_chunks"],
                "text_chunks": result["text_chunks"],
                "table_chunks": result["table_chunks"],
                "image_chunks": result["image_chunks"],
                "extraction_method": result["extraction_method"],
                "clause_summary": result["clause_summary"],
                "clauses": result["clauses"],
                "status": "ready"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    finally:
        # Always delete temp file regardless of success or failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Answers a question about an uploaded contract.

    Two modes:
    - use_agent=False: Direct RAG (faster, ~2-4s)
    - use_agent=True: LangGraph agent routing (smarter, ~4-8s)

    Caching: Same question on same session returns cached
    answer instantly. Cache TTL is 1 hour.

    Interview answer for Strides Q14 (latency reduction):
    "Three mechanisms: in-memory caching for repeated queries
    returning in under 10ms, singleton embedder pre-loaded at
    startup, and dual mode letting simple queries bypass the
    agent and save one LLM call."
    """

    # Input guardrail — before any processing
    is_valid, reason = check_input(request.question)
    if not is_valid:
        return QuestionResponse(
            answer=reason,
            sources=[],
            latency_seconds=0.0,
            status="blocked_by_guardrail"
        )

    # Check cache
    cache_key = get_cache_key(request.session_id, request.question)
    cached = get_from_cache(cache_key)
    if cached:
        return QuestionResponse(**cached)

    # Get answer
    if request.use_agent:
        response = get_agent_answer(
            question=request.question,
            session_id=request.session_id
        )
    else:
        response = get_answer(
            question=request.question,
            session_id=request.session_id
        )

    # Build response
    result = QuestionResponse(
        answer=response["answer"],
        sources=response.get("sources", []),
        confidence=response.get("confidence", 0.0),
        latency_seconds=response.get("latency_seconds", 0.0),
        status=response.get("status", "success"),
        warnings=response.get("warnings", []),
        cached=False
    )

    # Cache successful responses only
    if result.status == "success":
        set_cache(cache_key, result.model_dump())

    return result


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clears cache entries for a specific session.
    Called when user uploads a new contract to invalidate
    stale cached answers from the previous document.
    """
    keys_to_delete = [
        k for k, v in _cache.items()
        if session_id in str(v)
    ]
    for key in keys_to_delete:
        del _cache[key]

    return {
        "session_id": session_id,
        "cleared_entries": len(keys_to_delete)
    }