import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os


# Why 50 characters?
# A real text PDF page always has more than 50 chars.
# If we get less, the page is almost certainly a scanned image.
# This threshold is the same logic used in production OCR pipelines.
MIN_TEXT_LENGTH = 50


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extracts text from a PDF file.
    Automatically handles both text PDFs and scanned PDFs.

    Why two methods?
    - Text PDFs: PyMuPDF extracts directly — fast, accurate
    - Scanned PDFs: pytesseract OCR converts image to text — slower but handles real-world docs

    In legal, ~30% of contracts are scanned. Without OCR fallback,
    those contracts return empty strings and your RAG system fails silently.
    Silent failures are the worst kind — no error, just wrong answers.

    Returns:
        dict with pages, total_pages, doc_name, extraction_method, scanned_pages
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []
    scanned_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Try direct text extraction first — always faster
        text = page.get_text().strip()

        if len(text) >= MIN_TEXT_LENGTH:
            # Text PDF page — clean extraction
            pages.append({
                "page_num": page_num + 1,  # human readable, 1-indexed
                "text": text,
                "is_scanned": False
            })

        else:
            # Scanned page — route to OCR
            # Why 300 DPI?
            # Below 200 DPI, OCR accuracy drops sharply on small fonts.
            # 300 DPI is the industry standard for document OCR.
            # Higher DPI = better accuracy but slower and more memory.
            scanned_count += 1
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            ocr_text = pytesseract.image_to_string(img, lang="eng")

            pages.append({
                "page_num": page_num + 1,
                "text": ocr_text.strip(),
                "is_scanned": True
            })

    doc.close()

    # If more than half the pages needed OCR, classify as scanned document
    extraction_method = "ocr" if scanned_count > len(pages) / 2 else "text"

    return {
        "pages": pages,
        "total_pages": len(pages),
        "doc_name": os.path.basename(file_path),
        "extraction_method": extraction_method,
        "scanned_pages": scanned_count
    }


def get_full_text(extracted: dict) -> str:
    """
    Joins all page texts into one string with page markers.

    Why include [Page X] markers?
    When the LLM answers a question, it needs to cite the page number.
    If page markers are embedded in the text during extraction,
    they naturally appear in the retrieved chunks — so the LLM
    can always tell which page a piece of information came from.
    This is how you get reliable citations without extra complexity.
    """
    return "\n\n".join([
        f"[Page {p['page_num']}]\n{p['text']}"
        for p in extracted["pages"]
        if p["text"].strip()  # skip completely empty pages
    ])


def get_extraction_summary(extracted: dict) -> str:
    """
    Returns a human-readable summary of the extraction.
    Used for logging and debugging.
    """
    return (
        f"Document: {extracted['doc_name']} | "
        f"Pages: {extracted['total_pages']} | "
        f"Method: {extracted['extraction_method']} | "
        f"Scanned pages: {extracted['scanned_pages']}"
    )