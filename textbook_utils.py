import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np


def load_textbook(file_path: str) -> str:
    """Read PDF and return text."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into smaller chunks."""
    words = text.split()
    chunks, chunk = [], []
    for word in words:
        chunk.append(word)
        if len(chunk) >= chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vector, text):
        self.vectors.append(vector)
        self.texts.append(text)

    def query(self, query_vector, top_k=3):
        similarities = []
        for i, v in enumerate(self.vectors):
            sim = np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v))
            similarities.append((sim, self.texts[i]))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in similarities[:top_k]]


def index_textbook(chunks):
    """Create embeddings for chunks and store in vector DB."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    store = SimpleVectorStore()

    for chunk in chunks:
        vector = model.encode(chunk)
        store.add(vector, chunk)

    return model, store
