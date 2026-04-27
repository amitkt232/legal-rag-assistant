from dotenv import load_dotenv
import os

# Load .env file from project root
load_dotenv()

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Why Llama 3 8B and not 70B?
# 8B: fast response ~1-2 seconds, free tier, handles legal Q&A perfectly
# 70B: slower, hits rate limits faster, marginal quality gain for our use case
# In production self-hosted: we would use 70B on dedicated GPU
# Development model - fast, low token usage, tool calling supported
GROQ_MODEL = "llama-3.1-8b-instant"

# Production model - best quality (use for final demo and interviews)
GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"

# Why temperature 0.0?
# Legal answers must be deterministic and factual.
# Higher temperature = more creative = more hallucination risk.
# For a legal system, creativity is a bug not a feature.
# We want the same question to always get the same answer.
TEMPERATURE = 0.0

# Confidence threshold for retrieval
# If best chunk similarity score is below this, we don't answer.
# 0.3 chosen because sentence-transformer cosine similarity
# on legal text typically ranges 0.2 (unrelated) to 0.9 (near identical)
# Below 0.3 means retrieval found nothing genuinely relevant.
CONFIDENCE_THRESHOLD = 0.3

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Make sure .env file exists in project root with GROQ_API_KEY=your_key"
    )