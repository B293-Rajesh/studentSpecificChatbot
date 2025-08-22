import streamlit as st
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# PDF Text Loader
# -----------------------------
def load_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    if not text.strip():
        raise ValueError("No text found in PDF.")
    return text

# -----------------------------
# Chunk Text
# -----------------------------
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

# -----------------------------
# Simple Vector Store
# -----------------------------
class SimpleVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.metadata = []
        self.index = None

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
        results = [self.metadata[i] for i in I[0]]
        return results

# -----------------------------
# Index PDF
# -----------------------------
def index_pdf(pdf_file):
    text = load_pdf_text(pdf_file)
    chunks = chunk_text(text)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed_model.encode(chunks)
    store = SimpleVectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return embed_model, store, chunks

# -----------------------------
# Load LLM
# -----------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, llm

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Student Specific Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Specific Chatbot")
st.write("Upload your textbook (PDF) and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
user_input = st.text_input("Your question:")

if uploaded_file and user_input:
    # Index PDF
    embed_model, store, chunks = index_pdf(uploaded_file)

    # Load LLM
    tokenizer, llm = load_llm()

    # Get query embedding and search relevant chunks
    query_vec = embed_model.encode([user_input])[0]
    relevant_chunks = store.search(query_vec, k=3)
    context = "\n".join(relevant_chunks)

    # Generate answer
    prompt = f"Answer the question based on the following context:\n{context}\nQuestion: {user_input}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    # Display answer
    st.write("🧠 Answer")
    st.write(answer)
