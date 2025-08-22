import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ---------- PDF -> text ----------
def load_textbook_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


# ---------- chunking ----------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 60) -> list[str]:
    """Word-based chunks with overlap for continuity."""
    words = re.split(r"\s+", text.strip())
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)
    return chunks


# ---------- tiny in-memory vector store (cosine on unit vectors) ----------
class SimpleVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: list[np.ndarray] = []   # unit-normalized
        self._texts: list[str] = []

    def add(self, vec: np.ndarray, text: str):
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0.0:
            return
        self._vecs.append(v / n)            # store unit vectors
        self._texts.append(text)

    def search(self, q: np.ndarray, k: int = 3):
        if not self._vecs:
            return []
        q = np.asarray(q, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq == 0.0:
            return []
        q = q / nq
        E = np.stack(self._vecs, axis=0)    # [N, D], unit-normed
        sims = E @ q                         # cosine = dot of unit vectors
        top = np.argsort(sims)[::-1][:k]
        return [(self._texts[i], float(sims[i])) for i in top]


# ---------- build embeddings + index ----------
def index_textbook(chunks: list[str]):
    """
    all-MiniLM-L6-v2 (384-dim) for compact, fast embeddings.
    No explicit device to avoid NotImplementedError on some hosts.
    """
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = emb_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    store = SimpleVectorStore(dim=vecs.shape[1])
    for v, ch in zip(vecs, chunks):
        store.add(v, ch)
    return emb_model, store


# ---------- query helper ----------
def retrieve_top_k(emb_model, store: SimpleVectorStore, query: str, k: int = 3):
    q = emb_model.encode(query, convert_to_numpy=True)
    return store.search(q, k=k)
