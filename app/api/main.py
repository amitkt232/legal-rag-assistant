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
_cache = {}
CACHE_TTL_SECONDS = 3600


def get_cache_key(session_id: str, question: str) -> str:
    raw = f"{session_id}:{question.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_from_cache(key: str):
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
    _cache[key] = {
        "timestamp": time.time(),
        "response": response
    }


# ── Lifespan ──────────────────────────────────────────────
# Why no pre-loading here?
# Render free tier has 512MB RAM limit.
# sentence-transformers + torch alone uses ~400MB.
# Pre-loading at startup causes OOM before first request.
# Lazy loading (load on first request) keeps startup RAM low
# and allows the service to start within the memory limit.
# In production with more RAM, re-enable pre-loading.

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Legal RAG API...")
    print("API ready. Embedder will load on first request.")
    yield
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
    Used as Kubernetes readiness probe and Render health check.
    Returns immediately without loading any ML models.
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
    Embedder loads on first call to this endpoint.
    Raw PDF never persisted — temp file deleted after ingestion.
    """

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 50MB"
        )

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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Answers a question about an uploaded contract.
    Checks cache first — cached responses return in under 10ms.
    """

    is_valid, reason = check_input(request.question)
    if not is_valid:
        return QuestionResponse(
            answer=reason,
            sources=[],
            latency_seconds=0.0,
            status="blocked_by_guardrail"
        )

    cache_key = get_cache_key(request.session_id, request.question)
    cached = get_from_cache(cache_key)
    if cached:
        return QuestionResponse(**cached)

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

    result = QuestionResponse(
        answer=response["answer"],
        sources=response.get("sources", []),
        confidence=response.get("confidence", 0.0),
        latency_seconds=response.get("latency_seconds", 0.0),
        status=response.get("status", "success"),
        warnings=response.get("warnings", []),
        cached=False
    )

    if result.status == "success":
        set_cache(cache_key, result.model_dump())

    return result


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clears cache entries for a specific session.
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