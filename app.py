# app.py
import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ------------------ PDF Text Extraction ------------------
def load_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    if not text.strip():
        raise ValueError("No text found in PDF.")
    return text

# ------------------ Text Chunking ------------------
import re
def chunk_text(text, max_tokens=100):  # smaller chunks for memory
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], []
    current_len = 0
    for sentence in sentences:
        wc = len(sentence.split())
        if current_len + wc > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = wc
        else:
            current_chunk.append(sentence)
            current_len += wc
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ------------------ Simple Vector Store ------------------
class SimpleVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.metadata = []

    def add(self, vectors, metas):
        for v, m in zip(vectors, metas):
            vec = np.array(v, dtype=np.float32)
            self.vectors.append(vec)
            self.metadata.append(m)
        if self.vectors:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.stack(self.vectors))

    def search(self, query_vector, k=1):  # keep k small for memory
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(query_vector, k)
        results = [(self.metadata[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results

# ------------------ Index PDF ------------------
@st.cache_data
def index_pdf(pdf_file):
    text = load_pdf_text(pdf_file)
    chunks = chunk_text(text)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed_model.encode(chunks)
    store = SimpleVectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return embed_model, store, chunks

# ------------------ Load Flan-T5 Model ------------------
@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Student Specific Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Specific Chatbot")
st.write("Upload your textbook (PDF) and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
user_question = st.text_input("Your question:")

if uploaded_file:
    try:
        embed_model, store, chunks = index_pdf(uploaded_file)
        tokenizer, llm = load_generator()

        if user_question:
            # Embed query
            query_vec = embed_model.encode([user_question])[0]
            results = store.search(query_vec, k=1)  # use top 1 chunk

            # Use top chunk as context
            context = results[0][0]

            prompt = f"Answer the question using the following context:\n{context}\nQuestion: {user_question}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = llm.generate(**inputs, max_new_tokens=150)
            answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            st.subheader("🧠 Answer")
            st.write(answer)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
