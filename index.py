from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from textbook_utils import process_pdf

def load_index(pdf_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunks = process_pdf(pdf_path)
    store = FAISS.from_texts(chunks, embeddings)
    return store, chunks
