import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str, ocr_enabled: bool = True) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF.
    Falls back to OCR for scanned/image-only pages.
    Returns list of {page, text, source} dicts.
    """
    pages = []
    filename = Path(pdf_path).name

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # OCR fallback for image-heavy or scanned pages
            if ocr_enabled and _needs_ocr(text):
                text = _ocr_page(page, page_num, filename)

            text = _clean_text(text)
            if text.strip():
                pages.append({
                    "page": page_num + 1,
                    "text": text,
                    "source": filename,
                    "pdf_path": pdf_path,
                })

        doc.close()

    except Exception as e:
        logger.error(f"Failed to extract {pdf_path}: {e}")

    return pages


def _needs_ocr(text: str, min_chars: int = 50) -> bool:
    """Return True if page has too little native text to be useful."""
    return len(text.strip()) < min_chars


def _ocr_page(page, page_num: int, filename: str) -> str:
    """Render page to image and run Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        logger.debug(f"OCR applied to {filename} page {page_num + 1}")
        return text
    except ImportError:
        logger.warning("pytesseract/PIL not installed; skipping OCR")
        return ""
    except Exception as e:
        logger.warning(f"OCR failed on page {page_num + 1}: {e}")
        return ""


def _clean_text(text: str) -> str:
    """Normalize whitespace, remove page headers/footers patterns."""
    # Remove lines that look like page numbers or running headers (short, isolated lines)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just numbers (page numbers) or very short isolated fragments
        if re.match(r"^\d+$", stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Sliding-window chunker over page text.
    Each chunk retains metadata: source filename, page number, chunk index.
    """
    chunks = []

    for page_data in pages:
        text = page_data["text"]
        words = text.split()
        step = max(1, chunk_size - overlap)

        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start: start + chunk_size]
            if len(chunk_words) < 30:  # skip tiny tail chunks
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_index": i,
                "pdf_path": page_data["pdf_path"],
            })

    return chunks
