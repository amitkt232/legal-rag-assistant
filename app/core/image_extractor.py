import fitz  # PyMuPDF - already installed
import pytesseract  # already installed
from PIL import Image  # already installed
import io
from typing import List, Dict


# Minimum image size to process
# Why 100x100?
# Smaller images are usually decorative elements, logos, or artifacts
# not meaningful content. Processing them wastes time and adds noise.
# A signature block or diagram is always larger than 100x100 pixels.
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100


def extract_images_from_pdf(file_path: str) -> List[Dict]:
    """
    Extracts images from a PDF and converts them to text using OCR.

    Why extract images in legal contracts?
    Legal contracts often contain:
    - Signature blocks (scanned signatures)
    - Company stamps and seals
    - Diagrams showing organizational structure
    - Scanned annexures and schedules
    - Charts showing payment timelines

    Without image extraction, all of this is invisible to your RAG system.
    An analyst asking "is there a company seal on this contract?" gets
    "I don't know" even though the seal is clearly on page 12.

    How it works:
    1. PyMuPDF identifies all embedded images in the PDF
    2. We filter out tiny decorative images
    3. For each meaningful image, we run pytesseract OCR
    4. OCR text is added to the chunk pool with image metadata

    Interview answer for "how did you handle image data":
    "We used PyMuPDF to extract embedded images from PDFs, filtered
    by minimum size to remove decorative elements, then ran pytesseract
    OCR on each image to convert visual content to searchable text.
    Each image's OCR output was treated as a separate chunk with metadata
    indicating it came from an image rather than text - this helps the
    LLM cite it correctly as 'visual content on page X'."

    Returns list of dicts with page_num, image_text, image_index
    """

    images_found = []
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_images() returns list of image references on this page
        # Each item: (xref, smask, width, height, bpc, colorspace, ...)
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        for img_idx, img_ref in enumerate(image_list):
            xref = img_ref[0]
            width = img_ref[2]
            height = img_ref[3]

            # Skip tiny images - decorative elements, bullets, icons
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            try:
                # Extract raw image bytes from PDF
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image for pytesseract
                img = Image.open(io.BytesIO(image_bytes))

                # Run OCR on the image
                # Why lang="eng"? Legal contracts are English
                # For multilingual: lang="eng+hin" for Hindi+English
                ocr_text = pytesseract.image_to_string(
                    img,
                    lang="eng",
                    config="--psm 6"
                    # psm 6 = assume uniform block of text
                    # better for document images than default psm 3
                )

                ocr_text = ocr_text.strip()

                # Only include if OCR found actual text
                # Empty images (logos, decorative) return empty string
                if len(ocr_text) > 20:
                    images_found.append({
                        "page_num": page_num + 1,
                        "image_index": img_idx,
                        "image_text": f"[Image content, Page {page_num + 1}]\n{ocr_text}",
                        "width": width,
                        "height": height,
                        "ocr_char_count": len(ocr_text)
                    })

            except Exception as e:
                # Never crash the pipeline for one bad image
                print(f"  [Warning] Image extraction error page {page_num + 1}: {e}")
                continue

    doc.close()
    return images_found


def get_image_summary(images: List[Dict]) -> str:
    """Summary for logging."""
    if not images:
        return "No meaningful images found"
    return (
        f"Found {len(images)} images with extractable text "
        f"across {len(set(i['page_num'] for i in images))} pages"
    )