import streamlit as st
import requests
import json
import time
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Legal Contract Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Configuration ─────────────────────────────────────
# Why call FastAPI instead of importing directly?
# Separation of concerns — UI layer should not know about
# the AI pipeline internals. It only calls the API.
# This means in production, the UI and API can be deployed
# on separate servers, scaled independently, and replaced
# without touching each other.
API_BASE = "http://localhost:8000"


# ── Helper functions ──────────────────────────────────────

def upload_contract(file) -> dict:
    """Uploads PDF to FastAPI and returns ingestion result."""
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"{API_BASE}/upload", files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
        return None


def ask_question(session_id: str, question: str, use_agent: bool) -> dict:
    """Sends question to FastAPI and returns answer."""
    payload = {
        "session_id": session_id,
        "question": question,
        "use_agent": use_agent
    }
    response = requests.post(f"{API_BASE}/ask", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "answer": "Error connecting to API. Make sure the server is running.",
            "status": "error",
            "sources": [],
            "latency_seconds": 0,
            "cached": False
        }


def check_api_health() -> bool:
    """Checks if FastAPI is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .clause-card {
        background: #f8f9fa;
        border-left: 4px solid #4a90d9;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .risk-card {
        background: #fff8f0;
        border-left: 4px solid #ff6b35;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .answer-box {
        background: #f8fffe;
        border: 1px solid #d0e8e4;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
    .cached-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .source-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "clauses" not in st.session_state:
    st.session_state.clauses = None
if "clause_summary" not in st.session_state:
    st.session_state.clause_summary = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_agent" not in st.session_state:
    st.session_state.use_agent = False


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Legal Contract Assistant")
    st.markdown("---")

    # API status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline — run: python main.py")
        st.stop()

    st.markdown("---")

    # File upload
    st.markdown("### 📄 Upload Contract")
    uploaded_file = st.file_uploader(
        "Choose a PDF contract",
        type=["pdf"],
        help="Upload any contract PDF — NDA, vendor agreement, employment contract"
    )

    if uploaded_file is not None:
        # Only re-ingest if new file uploaded
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Analysing contract..."):
                result = upload_contract(uploaded_file)

            if result:
                st.session_state.session_id = result["session_id"]
                st.session_state.doc_name = result["doc_name"]
                st.session_state.clauses = result.get("clauses", {})
                st.session_state.clause_summary = result.get("clause_summary", "")
                st.session_state.messages = []  # Clear chat on new upload
                st.success(f"✅ Contract ready")
                st.rerun()

    # Document info
    if st.session_state.doc_name:
        st.markdown("---")
        st.markdown("### 📋 Document")
        st.info(f"**{st.session_state.doc_name}**")

    st.markdown("---")

    # Mode toggle
    st.markdown("### ⚙️ Settings")
    st.session_state.use_agent = st.toggle(
        "Use AI Agent",
        value=st.session_state.use_agent,
        help="Agent routes questions to the best tool. Slower but smarter."
    )

    if st.session_state.use_agent:
        st.caption("🤖 Agent mode: routes to retrieve, summarise, or risk tools")
    else:
        st.caption("⚡ Direct RAG mode: faster, best for specific questions")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main content ──────────────────────────────────────────
st.markdown('<div class="main-header">⚖️ Legal Contract Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a contract and ask questions in plain English</div>', unsafe_allow_html=True)

# No document uploaded yet
if not st.session_state.session_id:
    st.info("👈 Upload a contract PDF from the sidebar to get started")

    # Feature overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Instant Summary")
        st.markdown("Get key clauses extracted automatically — termination, liability, governing law, payment terms")
    with col2:
        st.markdown("### 💬 Ask Anything")
        st.markdown("Ask questions in plain English and get cited answers with exact page references")
    with col3:
        st.markdown("### 🚩 Risk Flags")
        st.markdown("Unusual or one-sided clauses flagged automatically — auto-renewal traps, unlimited liability")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Q&A Chat", "📋 Contract Summary"])


# ── Tab 1: Q&A Chat ───────────────────────────────────────
with tab1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])

                # Show metadata if available
                if "metadata" in message:
                    meta = message["metadata"]
                    cols = st.columns([2, 2, 2, 4])

                    with cols[0]:
                        status_color = "🟢" if meta["status"] == "success" else "🟡"
                        st.caption(f"{status_color} {meta['status']}")

                    with cols[1]:
                        st.caption(f"⏱️ {meta['latency']}s")

                    with cols[2]:
                        if meta.get("cached"):
                            st.markdown('<span class="cached-badge">⚡ cached</span>', unsafe_allow_html=True)

                    with cols[3]:
                        if meta.get("sources"):
                            pages = list(set(
                                s["page"] for s in meta["sources"]
                                if s.get("page")
                            ))
                            if pages:
                                pages_str = " ".join([
                                    f'<span class="source-badge">Page {p}</span>'
                                    for p in sorted(pages)[:5]
                                ])
                                st.markdown(pages_str, unsafe_allow_html=True)

                    if meta.get("warnings"):
                        for w in meta["warnings"]:
                            st.warning(f"⚠️ {w}")
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the contract..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching contract..."):
                response = ask_question(
                    st.session_state.session_id,
                    prompt,
                    st.session_state.use_agent
                )

            answer = response.get("answer", "No answer returned")
            st.markdown(answer)

            # Show metadata
            meta = {
                "status": response.get("status", "unknown"),
                "latency": response.get("latency_seconds", 0),
                "cached": response.get("cached", False),
                "sources": response.get("sources", []),
                "warnings": response.get("warnings", [])
            }

            cols = st.columns([2, 2, 2, 4])
            with cols[0]:
                status_color = "🟢" if meta["status"] == "success" else "🟡"
                st.caption(f"{status_color} {meta['status']}")
            with cols[1]:
                st.caption(f"⏱️ {meta['latency']}s")
            with cols[2]:
                if meta["cached"]:
                    st.markdown('<span class="cached-badge">⚡ cached</span>', unsafe_allow_html=True)
            with cols[3]:
                if meta["sources"]:
                    pages = list(set(
                        s["page"] for s in meta["sources"]
                        if s.get("page")
                    ))
                    if pages:
                        pages_str = " ".join([
                            f'<span class="source-badge">Page {p}</span>'
                            for p in sorted(pages)[:5]
                        ])
                        st.markdown(pages_str, unsafe_allow_html=True)

            if meta["warnings"]:
                for w in meta["warnings"]:
                    st.warning(f"⚠️ {w}")

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metadata": meta
        })

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("#### 💡 Suggested questions")
        suggestions = [
            "What is the termination clause?",
            "What are the payment terms?",
            "Which state law governs this contract?",
            "What are the risks in this contract?",
            "Who are the parties to this agreement?",
            "What is the liability cap?",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": suggestion
                    })
                    st.rerun()


# ── Tab 2: Contract Summary ───────────────────────────────
with tab2:
    if not st.session_state.clauses:
        st.info("Upload a contract to see the structured summary")
    else:
        clauses = st.session_state.clauses

        # Contract type and parties
        col1, col2 = st.columns(2)
        with col1:
            contract_type = clauses.get("contract_type", "Unknown")
            st.markdown(f"**Contract Type:** {contract_type}")
        with col2:
            parties = clauses.get("parties", "Not specified")
            st.markdown(f"**Parties:** {parties}")

        st.markdown("---")

        # Key clauses
        st.markdown("### 📌 Key Clauses")

        clause_fields = [
            ("termination_clause", "termination_page", "🔴 Termination"),
            ("liability_cap", "liability_page", "💰 Liability Cap"),
            ("governing_law", "governing_law_page", "⚖️ Governing Law"),
            ("payment_terms", "payment_page", "💳 Payment Terms"),
            ("renewal_terms", "renewal_page", "🔄 Renewal Terms"),
        ]

        found_any = False
        for field, page_field, label in clause_fields:
            value = clauses.get(field)
            page = clauses.get(page_field)
            if value:
                found_any = True
                page_ref = f" — Page {page}" if page else ""
                st.markdown(
                    f'<div class="clause-card">'
                    f'<strong>{label}{page_ref}</strong><br>{value}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        if not found_any:
            st.info("No key clauses extracted. Try asking in the Q&A tab.")

        # Risk flags
        risk_flags = clauses.get("risk_flags", [])
        if risk_flags:
            st.markdown("---")
            st.markdown("### 🚩 Risk Flags")
            st.caption("These clauses may need legal attention")
            for flag in risk_flags:
                st.markdown(
                    f'<div class="risk-card">⚠️ {flag}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("---")
            st.success("✅ No significant risk flags detected")

        # Raw summary
        with st.expander("📄 Full text summary"):
            st.text(st.session_state.clause_summary)