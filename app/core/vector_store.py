from langchain_community.vectorstores import Chroma
# from langchain.schema import Document
from langchain_core.documents import Document
from app.core.embedder import get_embedder
from typing import List, Optional
import os


# Why ChromaDB?
# 1. Runs as a local file — no server to manage, no cloud account
# 2. Persists to disk — data survives restarts
# 3. Supports metadata filtering — critical for session isolation
# 4. Free and open source — matches our privacy-first architecture
#
# In production at scale: we would migrate to Pinecone (managed)
# or Weaviate (self-hosted cluster). ChromaDB handles up to
# ~1M vectors comfortably on a single machine.

CHROMA_BASE_PATH = "chroma_db"


def create_vector_store(
    chunks: List[Document],
    session_id: str
) -> Chroma:
    """
    Creates a ChromaDB collection for a specific session.

    Why session-scoped collections?
    If two lawyers upload different contracts simultaneously,
    their chunks must never mix. A question about Contract A
    must only retrieve chunks from Contract A.

    We achieve this by using session_id as the ChromaDB
    collection name. Each upload gets its own isolated namespace.

    This is a critical production concern — in a multi-user
    system, data isolation is a security requirement, not just
    a nice-to-have.
    """

    embedder = get_embedder()

    # Each session gets its own folder in chroma_db/
    persist_path = os.path.join(CHROMA_BASE_PATH, session_id)

    # Chroma.from_documents does 3 things in one call:
    # 1. Embeds each chunk using our local embedder
    # 2. Stores the vector + original text + metadata
    # 3. Persists everything to disk at persist_path
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_path,
        collection_name=f"session_{session_id}"
    )

    return db


def load_vector_store(session_id: str) -> Optional[Chroma]:
    """
    Loads an existing ChromaDB collection for a session.

    Called every time a user asks a question —
    we do not re-embed the document on each query.
    Ingestion happens once. Retrieval happens many times.
    This separation is a key performance decision.
    """

    persist_path = os.path.join(CHROMA_BASE_PATH, session_id)

    if not os.path.exists(persist_path):
        return None  # session not found

    embedder = get_embedder()

    db = Chroma(
        persist_directory=persist_path,
        embedding_function=embedder,
        collection_name=f"session_{session_id}"
    )

    return db


def get_retriever(session_id: str):
    """
    Returns a configured retriever for a session.

    Why MMR (Maximal Marginal Relevance)?
    Default similarity search returns the top 5 most similar chunks.
    Problem: if the same clause appears 3 times in the document
    (e.g. termination is mentioned in clause 4, clause 8, and appendix),
    you get 3 nearly identical chunks — wasted context window.

    MMR balances relevance AND diversity:
    - Fetches top 20 candidates by similarity
    - Picks 5 that are both relevant AND different from each other
    - Result: richer context for the LLM, better answers

    This is a retrieval failure mode you discovered from your
    escalation experience — "retrieved documents have the answer
    but too much noise" — exactly what your notebook mentioned.
    MMR solves it.
    """

    db = load_vector_store(session_id)

    if db is None:
        raise ValueError(f"No vector store found for session: {session_id}")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,          # return 5 chunks to LLM
            "fetch_k": 20,   # consider 20 candidates first
            "lambda_mult": 0.7
            # lambda_mult: 0 = max diversity, 1 = max relevance
            # 0.7 = slightly favour relevance over diversity
            # tuned on our test set in evaluation phase
        }
    )

    return retriever