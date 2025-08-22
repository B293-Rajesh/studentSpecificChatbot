# textbook_utils.py
import os
from typing import Any
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline


# -------------------------------
# PDF utilities
# -------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Try to extract text using pypdf (works for text-based PDFs)."""
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
        raise RuntimeError(f"OCR fallback failed: {e}")


# -------------------------------
# QA Chain builder
# -------------------------------

def build_qa_chain(store: Any, model_id: str = "google/flan-t5-base") -> RetrievalQA:
    """
    Build a LangChain RetrievalQA chain using a HuggingFace seq2seq model.
    - store: FAISS vector store
    - model_id: HuggingFace model ID (default: flan-t5-base)
    """
    # Load HF model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Wrap in HuggingFace pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Build RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False,
    )
    return qa
