from app.core.pdf_extractor import extract_text_from_pdf, get_extraction_summary
from app.core.chunker import create_chunks, get_chunk_stats
from app.core.vector_store import create_vector_store
import uuid
import os


def ingest_contract(file_path: str) -> dict:
    """
    Complete ingestion pipeline for a single contract PDF.

    This is the function you describe in interviews:
    "When a user uploads a contract, it goes through our
    ingestion pipeline — extraction, chunking, embedding,
    and storage — in under 30 seconds. After that, the
    document is fully searchable."

    Steps:
    1. Extract text (handles text PDF and scanned PDF)
    2. Create chunks with metadata
    3. Embed and store in ChromaDB
    4. Return session_id for the user

    The session_id is the key that connects a user to
    their uploaded document for all future queries.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Generate unique session ID for this upload
    session_id = str(uuid.uuid4())

    print(f"\n{'='*50}")
    print(f"Starting ingestion pipeline")
    print(f"Session ID: {session_id}")
    print(f"{'='*50}")

    # Step 1: Extract text from PDF
    print("\n[Step 1/3] Extracting text from PDF...")
    extracted = extract_text_from_pdf(file_path)
    print(f"  {get_extraction_summary(extracted)}")

    # Step 2: Create chunks with metadata
    print("\n[Step 2/3] Chunking document...")
    chunks = create_chunks(
        extracted_pages=extracted["pages"],
        doc_name=extracted["doc_name"],
        session_id=session_id
    )
    stats = get_chunk_stats(chunks)
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Avg chars per chunk: {stats['avg_chars_per_chunk']}")
    print(f"  Pages covered: {stats['pages_covered']}")

    # Step 3: Embed and store
    print("\n[Step 3/3] Embedding and storing in ChromaDB...")
    print("  (First run downloads ~90MB model — subsequent runs are instant)")
    db = create_vector_store(chunks, session_id)
    print(f"  Stored {len(chunks)} vectors in ChromaDB")

    print(f"\n{'='*50}")
    print(f"Ingestion complete!")
    print(f"Session ID: {session_id}")
    print(f"{'='*50}\n")

    return {
        "session_id": session_id,
        "doc_name": extracted["doc_name"],
        "total_pages": extracted["total_pages"],
        "total_chunks": len(chunks),
        "extraction_method": extracted["extraction_method"],
        "avg_chunk_chars": stats["avg_chars_per_chunk"],
        "status": "ready"
    }