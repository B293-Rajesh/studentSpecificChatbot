import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from textbook_utils import process_pdf

def load_index(pdf_path):
    # Load embeddings model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Extract chunks
    chunks = process_pdf(pdf_path)
    if not chunks:
        raise ValueError(f"No text chunks extracted from {pdf_path}")
    
    # Encode
    embeddings = embedder.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    )
    
    # Ensure embeddings are 2D
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return {"index": index, "chunks": chunks, "embedder": embedder}, chunks
