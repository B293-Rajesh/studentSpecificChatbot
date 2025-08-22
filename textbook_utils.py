# textbook_utils.py
import os
import io
from typing import Optional
from pypdf import PdfReader
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract

def extract_text_from_pdf(pdf_path: str) -> str:
    """Try to extract text using pypdf (fast, works for selectable text)."""
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            text_parts.append(txt)
    return "\n".join(text_parts)

def ocr_pdf(pdf_path: str) -> str:
    """
    Fallback OCR using pdf2image + pytesseract.
    Requires system packages: poppler (for pdf2image) and tesseract.
    """
    # If pdf_path is bytes stream, use convert_from_bytes; here we assume path
    images = convert_from_path(pdf_path, dpi=200)
    text_parts = []
    for img in images:
        txt = pytesseract.image_to_string(img)
        text_parts.append(txt)
    return "\n".join(text_parts)

def load_pdf_text_or_ocr(pdf_path: str) -> str:
    """
    Attempt to extract text; if empty, attempt OCR.
    Returns the full extracted text (may be empty).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if os.path.getsize(pdf_path) == 0:
        raise ValueError(f"PDF is empty: {pdf_path}")

    # 1) try normal text extraction
    text = extract_text_from_pdf(pdf_path)
    if text and text.strip():
        return text

    # 2) fallback to OCR (may be slow)
    try:
        text_ocr = ocr_pdf(pdf_path)
        if text_ocr and text_ocr.strip():
            return text_ocr
        else:
            return ""
    except Exception as e:
        # bubble up a helpful error
        raise RuntimeError(f"OCR fallback failed: {e}")
