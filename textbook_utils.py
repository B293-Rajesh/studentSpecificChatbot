import fitz  # PyMuPDF
import re
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

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

# STEP 3: Build FAISS vector store using LangChain
def index_pdf(pdf_path):
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # HuggingFace embeddings wrapper
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build FAISS index
    store = FAISS.from_documents(docs, embedder)
    
    return chunks, store
