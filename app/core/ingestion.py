from app.core.pdf_extractor import extract_text_from_pdf, get_extraction_summary, get_full_text
from app.core.table_extractor import extract_tables_from_pdf, get_table_summary
from app.core.image_extractor import extract_images_from_pdf, get_image_summary
from app.core.chunker import create_chunks, get_chunk_stats
from app.core.vector_store import create_vector_store
from app.core.clause_extractor import extract_clauses, format_clause_summary
from langchain_core.documents import Document
import uuid
import os


def ingest_contract(file_path: str) -> dict:
    """
    Complete ingestion pipeline - now handles text, tables, and images.

    Interview answer for "explain your ingestion pipeline":
    "Our ingestion pipeline has 5 steps. First, text extraction using
    PyMuPDF with OCR fallback for scanned pages. Second, table extraction
    using pdfplumber which preserves row and column structure and converts
    tables to markdown format. Third, image extraction using PyMuPDF's
    image API with pytesseract OCR to convert visual content to text.
    Fourth, all three content types are chunked and embedded using
    sentence-transformers locally - no data leaves the machine. Fifth,
    we run structured clause extraction using the LLM with Pydantic
    output schema to generate an instant contract summary."

    Why combine all three into one pipeline?
    In a real legal contract, the answer to "what are the payment terms"
    might be in a paragraph of text on page 3, a table on page 8, or
    an image of a scanned schedule on page 15. If you only extract text,
    you miss two of the three possible locations.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    session_id = str(uuid.uuid4())

    print(f"\n{'='*55}")
    print(f"INGESTION PIPELINE STARTED")
    print(f"Session: {session_id}")
    print(f"{'='*55}")

    # ── Step 1: Extract text ──────────────────────────────────
    print("\n[1/5] Extracting text...")
    extracted = extract_text_from_pdf(file_path)
    print(f"  {get_extraction_summary(extracted)}")

    # ── Step 2: Extract tables ────────────────────────────────
    print("\n[2/5] Extracting tables...")
    tables = extract_tables_from_pdf(file_path)
    print(f"  {get_table_summary(tables)}")

    # ── Step 3: Extract images ────────────────────────────────
    print("\n[3/5] Extracting images...")
    images = extract_images_from_pdf(file_path)
    print(f"  {get_image_summary(images)}")

    # ── Step 4: Chunk everything and store ────────────────────
    print("\n[4/5] Chunking and embedding...")

    # Text chunks from Day 2
    all_chunks = create_chunks(
        extracted_pages=extracted["pages"],
        doc_name=extracted["doc_name"],
        session_id=session_id
    )

    # Table chunks - each table becomes its own document
    # Why not mix tables into text chunks?
    # Tables have a different structure. Mixing them breaks
    # the markdown formatting that makes them readable.
    for table in tables:
        table_doc = Document(
            page_content=table["table_text"],
            metadata={
                "chunk_id": str(uuid.uuid4()),
                "session_id": session_id,
                "doc_name": extracted["doc_name"],
                "page_num": table["page_num"],
                "content_type": "table",  # signal for debugging
                "source": f"{extracted['doc_name']}_table_{table['table_index']}"
            }
        )
        all_chunks.append(table_doc)

    # Image chunks - each image's OCR text becomes its own document
    for image in images:
        image_doc = Document(
            page_content=image["image_text"],
            metadata={
                "chunk_id": str(uuid.uuid4()),
                "session_id": session_id,
                "doc_name": extracted["doc_name"],
                "page_num": image["page_num"],
                "content_type": "image",  # signal for debugging
                "source": f"{extracted['doc_name']}_image_{image['image_index']}"
            }
        )
        all_chunks.append(image_doc)

    stats = get_chunk_stats(all_chunks)
    print(f"  Text chunks: {stats['total_chunks'] - len(tables) - len(images)}")
    print(f"  Table chunks: {len(tables)}")
    print(f"  Image chunks: {len(images)}")
    print(f"  Total chunks: {stats['total_chunks']}")

    # Embed and store all chunks
    create_vector_store(all_chunks, session_id)
    print(f"  Stored in ChromaDB")

    # ── Step 5: Extract structured clauses ───────────────────
    print("\n[5/5] Extracting structured clauses...")
    full_text = get_full_text(extracted)
    clauses = extract_clauses(full_text)
    print(f"  Contract type: {clauses.get('contract_type', 'Unknown')}")
    print(f"  Risk flags found: {len(clauses.get('risk_flags', []))}")

    print(f"\n{'='*55}")
    print(f"INGESTION COMPLETE")
    print(f"Session ID: {session_id}")
    print(f"{'='*55}\n")

    return {
        "session_id": session_id,
        "doc_name": extracted["doc_name"],
        "total_pages": extracted["total_pages"],
        "total_chunks": stats["total_chunks"],
        "text_chunks": stats["total_chunks"] - len(tables) - len(images),
        "table_chunks": len(tables),
        "image_chunks": len(images),
        "extraction_method": extracted["extraction_method"],
        "clauses": clauses,
        "clause_summary": format_clause_summary(clauses),
        "status": "ready"
    }