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
    results = store.similarity_search(query, k=top_k)
    return [res.page_content for res in results]
