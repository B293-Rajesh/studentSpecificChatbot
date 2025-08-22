import fitz  # PyMuPDF

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_chunks.extend(text.split("\n"))
    return [chunk.strip() for chunk in text_chunks if chunk.strip()]

def query_index(store, query, top_k=3):
    embedder = store["embedder"]
    index = store["index"]
    chunks = store["chunks"]
    
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(query_vec, top_k)
    
    return [chunks[i] for i in ids[0]]
