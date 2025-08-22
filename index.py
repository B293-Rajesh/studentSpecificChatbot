import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from textbook_utils import process_pdf

def load_index(pdf_path):
    # Load embeddings model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Extract chunks from textbook
    chunks = process_pdf(pdf_path)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return {"index": index, "chunks": chunks, "embedder": embedder}, chunks
