from myvectorstore.index import SimpleVectorStore
from sentence_transformers import SentenceTransformer
import os
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

# STEP 2: Index chunks
def index_textbook(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)
    store = SimpleVectorStore(dim=384)
    store.add(vectors, chunks)
    return model, store

# STEP 3: Search for a query
def query_textbook(model, store, question):
    query_vec = model.encode([question])[0]
    results = store.search(query_vec, k=3)
    print(f"\n Top matches for: '{question}'\n")
    for i, (text, dist) in enumerate(results, 1):
        print(f"{i}. [{dist:.4f}] {text}\n")

# MAIN
if __name__ == "__main__":
    filepath = "x_biology_em.pdf"  # Change this
    question = "What is respiration?"

    text = load_textbook_pdf(filepath)
    chunks = chunk_text(text)
    model, store = index_textbook(chunks)
    query_textbook(model, store, question)
