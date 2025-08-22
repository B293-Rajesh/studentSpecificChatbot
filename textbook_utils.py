# textbook_utils.py
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------- PDF Loader -----------------
def load_textbook_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# ----------------- Chunking -----------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# ----------------- Vector Store -----------------
class SimpleVectorStore:
    def __init__(self, dim):
        self.vectors = []
        self.chunks = []
        self.dim = dim

    def add(self, vector, chunk):
        self.vectors.append(vector)
        self.chunks.append(chunk)

    def search(self, query_vector, top_k=3):
        scores = []
        for i, vec in enumerate(self.vectors):
            sim = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
            scores.append((self.chunks[i], float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ----------------- Indexing -----------------
def index_textbook(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # force CPU
    store = SimpleVectorStore(dim=384)
    for chunk in chunks:
        vec = model.encode(chunk)
        store.add(vec, chunk)
    return model, store

# ----------------- Querying -----------------
def query_textbook(model, store, query, top_k=3):
    query_vec = model.encode(query)
    return store.search(query_vec, top_k=top_k)
