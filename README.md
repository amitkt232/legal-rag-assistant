# ⚖️ Legal Contract Intelligence Assistant

An AI-powered legal contract review system built with RAG, LangGraph agents, and open-source LLMs. Upload any contract PDF and get instant clause extraction, risk flagging, and natural language Q&A with page-level citations.

---

## 🎯 Business Problem

Legal analysts spend 2–3 hours reviewing each contract manually. They frequently miss auto-renewal clauses, unusual liability terms, and one-sided indemnity provisions — costing companies significant money.

**This system reduces contract review time from 3 hours to under 20 minutes.**

---

## 🏗️ Architecture
PDF Upload → Extraction Pipeline → Vector Store → RAG Chain / Agent → FastAPI → Streamlit UI

**Two separate pipelines:**
- **Ingestion** (runs once per upload): PDF extraction → chunking → embedding → ChromaDB storage
- **Query** (runs per question): retrieval → context assembly → LLM generation → guardrail validation

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-format PDF extraction | Text PDFs + scanned PDFs via OCR + tables + images |
| 🔍 Hybrid retrieval | Vector search + metadata filtering + MMR diversity |
| 🤖 LangGraph agent | Routes questions to retrieve, summarise, or risk-analysis tools |
| 📋 Structured clause extraction | Auto-extracts termination, liability, governing law, payment terms |
| 🚩 Risk flagging | Detects unusual or one-sided clauses automatically |
| 🛡️ Guardrails | Input validation (prompt injection, off-topic) + output validation (PII, citations) |
| ⚡ Response caching | In-memory cache with 1hr TTL — repeated queries served instantly |
| 📊 RAGAS evaluation | 20-question golden dataset — faithfulness: 0.667, relevancy: 0.741 |
| 🧪 Pytest API tests | 6 tests covering upload, Q&A, guardrails, and caching |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Llama 3 via Groq API | Open source, fast, free tier, tool-calling support |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local — no data leaves the machine |
| Vector DB | ChromaDB | Local, session-isolated, no cloud required |
| PDF extraction | PyMuPDF + pytesseract | Handles text and scanned PDFs |
| Table extraction | pdfplumber | Preserves row/column structure as markdown |
| Agent framework | LangGraph 1.x | Current standard, replaced AgentExecutor |
| RAG framework | LangChain 1.x | LCEL chains, modular architecture |
| API | FastAPI 0.136 | Industry standard for ML APIs |
| UI | Streamlit | Demo layer for rapid prototyping |
| Evaluation | RAGAS 0.4.3 | Faithfulness + AnswerRelevancy metrics |
| Testing | pytest + httpx | Async API testing |

**Why open source stack?**
Legal contracts contain confidential client data. Using a fully local stack (local embeddings, local vector DB) means contract text never leaves the organisation's infrastructure. Only the Groq LLM API call is external — in a full production deployment this would be replaced with a self-hosted Llama 3 instance.

---

## 📁 Project Structure
legal-rag-assistant/
│
├── app/
│   ├── core/
│   │   ├── pdf_extractor.py      # Text + OCR extraction
│   │   ├── table_extractor.py    # pdfplumber table extraction
│   │   ├── image_extractor.py    # PyMuPDF image OCR
│   │   ├── chunker.py            # Metadata-rich chunking
│   │   ├── embedder.py           # Singleton sentence-transformer
│   │   ├── vector_store.py       # ChromaDB with session isolation
│   │   ├── ingestion.py          # Complete ingestion pipeline
│   │   ├── prompt.py             # Legal-specific prompt template
│   │   ├── qa_chain.py           # RAG chain with confidence gating
│   │   ├── agent_tools.py        # 3 retrieval tools for agent
│   │   ├── agent.py              # LangGraph ReAct agent
│   │   ├── guardrails.py         # Input + output validation
│   │   ├── clause_extractor.py   # Structured clause extraction
│   │   └── config.py             # Model and threshold config
│   └── api/
│       └── main.py               # FastAPI endpoints
│
├── evaluation/
│   ├── golden_dataset.json       # 20 human-authored Q&A pairs
│   └── run_evaluation.py         # RAGAS evaluation pipeline
│
├── tests/
│   └── test_api.py               # 6 pytest API tests
│
├── streamlit_app.py              # Streamlit demo UI
├── main.py                       # FastAPI entry point
├── requirements.txt              # All dependencies
└── .env.example                  # Environment variables template


---

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/legal-rag-assistant.git
cd legal-rag-assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get free key at: https://console.groq.com
```

**5. Run the FastAPI backend**
```bash
python main.py
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**6. Run the Streamlit UI (new terminal)**
```bash
streamlit run streamlit_app.py
# UI at http://localhost:8501
```

---

## 🧪 Running Tests

```bash
pytest tests/test_api.py -v
```

Expected output:
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_upload_valid_pdf PASSED
tests/test_api.py::test_upload_non_pdf_rejected PASSED
tests/test_api.py::test_ask_valid_question PASSED
tests/test_api.py::test_guardrail_blocks_injection PASSED
tests/test_api.py::test_caching_works PASSED
6 passed in ~100s

---

## 📊 Evaluation Results

Evaluated on 20 human-authored questions across 8 clause categories using RAGAS 0.4.3 with Llama 3 as judge LLM.

| Metric | Score | Target |
|---|---|---|
| Faithfulness | 0.667 | > 0.80 |
| Answer Relevancy | 0.741 | > 0.75 |
| Success Rate | 100% | 100% |
| Avg Latency | 5.85s | < 5s |

**Strong categories:** Governing law (1.000), Parties (1.000), Intellectual property (1.000), Payment (0.917)

**Improvement areas:** Subcontracting (0.000 — retrieval miss), Insurance (0.250 — wrong chunk retrieved)

**Root cause analysis:**
- Subcontracting clause retrieval miss — query "Can vendor subcontract without permission?" did not semantically match the chunk containing the answer. Fix: MultiQueryRetriever generating 3 query reformulations.
- Insurance clause — correct chunk exists but retrieved wrong schedule section. Fix: larger chunk overlap for dense schedule sections.

**Planned improvements:**
- MultiQueryRetriever for paraphrase-sensitive queries (estimated +0.15 faithfulness)
- Larger chunk overlap for Schedule sections (estimated +0.10 faithfulness)
- Re-evaluate with 70B model as RAGAS judge for more accurate scoring

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check — used as Kubernetes readiness probe |
| POST | `/upload` | Upload and ingest a contract PDF (max 50MB) |
| POST | `/ask` | Ask a question — supports direct RAG or agent mode |
| DELETE | `/session/{id}` | Clear session cache on new upload |

Full interactive docs at `http://localhost:8000/docs`

### Example request — upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@contract.pdf"
```

### Example request — ask
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "question": "What is the termination clause?",
    "use_agent": false
  }'
```

### Example response
```json
{
  "answer": "Either party may terminate with 60 days written notice [Page 14].",
  "sources": [{"page": 14, "doc_name": "contract.pdf"}],
  "confidence": 0.82,
  "latency_seconds": 3.2,
  "status": "success",
  "warnings": [],
  "cached": false
}
```

---

## 🛡️ Security Design

- **Data privacy**: Raw PDFs never persisted — written to temp file, ingested to vectors, deleted immediately
- **Session isolation**: Each upload gets a UUID session with a separate ChromaDB collection — users cannot access each other's documents
- **Guardrails — input**: Prompt injection detection, off-topic filtering, length validation, HTML sanitisation
- **Guardrails — output**: Citation validation, PII pattern detection, short answer rejection
- **Local embeddings**: Contract text never sent to any external embedding API — sentence-transformers runs on CPU locally

---

## 🤖 Agent Architecture

The LangGraph ReAct agent routes questions to 3 specialised tools:

User question
↓
LangGraph ReAct Agent
(Llama 3 decides routing)
↓
┌─────────────────────────────────────┐
│ retrieve_and_answer                 │ ← specific clause questions
│ summarise_contract                  │ ← overview requests
│ flag_contract_risks                 │ ← risk analysis
└─────────────────────────────────────┘
↓
Raw text returned to agent LLM
Agent generates cited answer

**Why tools return raw text (not LLM-generated answers):**
Tools are data fetchers, not answer generators. Having tools call the LLM internally creates nested LLM calls — double token usage and doubled latency. The agent's own LLM generates the final answer from tool output. This is the correct agentic pattern.

---

## 📝 Known Issues and Roadmap

**Known issues:**
- Query sensitivity on paraphrase variations → fix: MultiQueryRetriever
- Groq free tier daily token limit affects heavy testing → fix: paid tier or self-hosted Llama 3
- LangGraph `create_react_agent` deprecation warning → fix: migrate to `langchain.agents.create_agent` in next sprint

**Roadmap:**
- [ ] MultiQueryRetriever for improved context recall
- [ ] Redis semantic caching for production scale
- [ ] Self-hosted Llama 3 for complete data residency
- [ ] Docker + GitHub Actions CI/CD pipeline
- [ ] Fine-tuning on legal domain terminology
- [ ] Conversation memory for multi-turn sessions

---

## 🏭 Production Considerations

| Concern | Current (Portfolio) | Production |
|---|---|---|
| LLM | Groq free API | Self-hosted Llama 3 on GPU cluster |
| Vector DB | Local ChromaDB | Pinecone or Weaviate cluster |
| Caching | In-memory dict | Redis with semantic similarity |
| Deployment | Local / Render | Kubernetes with HPA autoscaling |
| Monitoring | Print logs | LangSmith + Grafana + Prometheus |
| Auth | None | OAuth2 + RBAC per organisation |

---

## 👤 About This Project

Built to demonstrate end-to-end GenAI engineering skills:

- **RAG pipeline** — multi-format ingestion, hybrid retrieval, confidence gating, RAGAS evaluation
- **Agent design** — LangGraph ReAct agent with 3 tools, correct tool-as-data-fetcher pattern
- **Production API** — FastAPI with caching, lifespan events, async testing with pytest
- **Security** — guardrails, session isolation, temp file cleanup, PII detection
- **Engineering practices** — feature branch Git workflow, conventional commits, pytest CI

**Background:** 3.6 years in GenAI and escalation engineering — every design decision in this project was informed by real production failure patterns encountered while handling Sev escalations on GenAI systems.


