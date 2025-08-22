import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# STEP 1: Load PDF text
def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    if not text.strip():
        raise ValueError("No text found in PDF.")
    return text

# STEP 2: Chunk text
def chunk_text(text, max_tokens=200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], []
    current_len = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# STEP 3: Build vector store
class SimpleVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.metadata = []

    def add(self, vectors, metas):
        for v, m in zip(vectors, metas):
            vec = np.array(v, dtype=np.float32)
            self.vectors.append(vec)
            self.metadata.append(m)
        if self.vectors:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.stack(self.vectors))

    def search(self, query_vector, k=3):
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(query_vector, k)
        results = [(self.metadata[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results

def index_pdf(pdf_path):
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)
    store = SimpleVectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return model, store, chunks
