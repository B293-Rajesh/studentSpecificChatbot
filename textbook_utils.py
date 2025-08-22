import numpy as np
import faiss

def process_pdf(pdf_path):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text_chunks.extend(lines)
    return text_chunks

def query_index(store, query, top_k=3):
    """
    store = {"index": FAISS_index, "chunks": [chunks], "embedder": model}
    """
    embedder = store["embedder"]
    index = store["index"]
    chunks = store["chunks"]

    # Encode query
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)  # similarity search

    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(chunks):  # avoid out of range
            results.append((chunks[i], float(score)))
    return results
