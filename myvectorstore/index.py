import numpy as np
import pickle
import os

class SimpleVectorStore:
    def __init__(self, dim, index_path="vector_index.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.vectors = []
        self.metadata = []

    def add(self, vectors, metas):
        for v, m in zip(vectors, metas):
            vec = np.array(v, dtype=np.float32)
            if vec.shape[0] != self.dim:
                raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vec.shape[0]}")
            self.vectors.append(vec)
            self.metadata.append(m)
        self._save()

    def search(self, query_vector, k=3, metric="l2"):
        query_vector = np.array(query_vector, dtype=np.float32)
        if metric == "l2":
            dists = [np.linalg.norm(vec - query_vector) for vec in self.vectors]
        elif metric == "cosine":
            dists = [1 - np.dot(vec, query_vector) /
                     (np.linalg.norm(vec) * np.linalg.norm(query_vector) + 1e-10)
                     for vec in self.vectors]
        else:
            raise ValueError("Unsupported metric. Use 'l2' or 'cosine'.")

        sorted_idx = np.argsort(dists)
        top_k_idx = sorted_idx[:k]
        return [(self.metadata[i], dists[i]) for i in top_k_idx]

    def _save(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump((self.vectors, self.metadata), f)


# ---------------- UTILS ---------------- #
from sentence_transformers import SentenceTransformer
import fitz
import re

# STEP 1: Load and chunk PDF
def load_textbook_pdf(filepath):
    doc = fitz.open(filepath)
    text = "".join([page.get_text() for page in doc])
    return text

def chunk_text(text, max_tokens=100):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# STEP 2: Index chunks (always overwrite index)
def index_textbook(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)
    
    # ✅ Always create fresh index (ignore old pickle)
    if os.path.exists("vector_index.pkl"):
        os.remove("vector_index.pkl")
    
    store = SimpleVectorStore(dim=384)
    store.add(vectors, chunks)
    return model, store

# STEP 3: Search for a query
def query_textbook(model, store, question):
    query_vec = model.encode([question])[0]
    results = store.search(query_vec, k=3)
    return results
