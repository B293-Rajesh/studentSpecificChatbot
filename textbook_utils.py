import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from index import SimpleVectorStore
import PyPDF2

def load_textbook(file_path):
    """
    Load textbook from .txt or .pdf file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    else:
        raise ValueError("Unsupported file format. Use .txt or .pdf")

def chunk_text(text, chunk_size=500):
    """
    Split text into chunks of approximately `chunk_size` words.
    """
    words = re.split(r"\s+", text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def index_textbook(chunks):
    """
    Build vector index from textbook chunks.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode(chunks, show_progress_bar=True)

    store = SimpleVectorStore()
    for chunk, emb in zip(chunks, embeddings):
        store.add(emb, chunk)

    return model, store
