import os
import streamlit as st
from textbook_utils import (
    load_textbook_pdf,
    chunk_text,
    index_textbook,
    retrieve_top_k,
)
from transformers import pipeline

PDF_PATH = os.getenv("PDF_PATH", "x_biology_em.pdf")  # your PDF file at repo root
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "google/flan-t5-base")  # simple, public model
TOP_K_DEFAULT = 3

st.set_page_config(page_title="🎓 Student Specific Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Specific Chatbot")
st.caption("RAG over your textbook → simple generator (FLAN-T5) for answers")

# ---------- Cache: index the PDF once ----------
@st.cache_resource(show_spinner="Indexing textbook (first run only)…")
def build_index(pdf_path: str):
    text = load_textbook_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=500, overlap=60)
    emb_model, store = index_textbook(chunks)
    return emb_model, store, chunks

# ---------- Cache: load a simple, public generator ----------
@st.cache_resource(show_spinner="Loading generator…")
def load_generator(model_id: str):
    # text2text works great with FLAN-T5
    return pipeline(
        task="text2text-generation",
        model=model_id,
        device=-1,                  # CPU (works on Spaces without GPU)
        max_new_tokens=256
    )

def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[Passage {i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the user question using ONLY the information in the CONTEXT. "
        "If the context is insufficient, say you don't have enough information.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )

def generate_answer(generator, question: str, contexts: list[str]) -> str:
    if not contexts:
        return "I couldn't find relevant context in the textbook to answer that."
    prompt = build_prompt(question, contexts)
    out = generator(prompt, num_beams=4, do_sample=False)
    text = out[0]["generated_text"].strip()
    return text

# ---------- Build index ----------
try:
    emb_model, store, all_chunks = build_index(PDF_PATH)
except FileNotFoundError:
    st.error(f"PDF not found at `{PDF_PATH}`. Upload it to the repo root or set env var PDF_PATH.")
    st.stop()

# ---------- Load generator ----------
generator = load_generator(GEN_MODEL_ID)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider("Top-K passages", 1, 7, TOP_K_DEFAULT, 1)
    show_ctx = st.checkbox("Show retrieved passages", value=True)

# ---------- Main UI ----------
user_q = st.text_input("Your question:")
if st.button("Ask") and user_q.strip():
    with st.spinner("Retrieving…"):
        hits = retrieve_top_k(emb_model, store, user_q, k=top_k)

    if not hits:
        st.warning("No relevant passages found in the textbook.")
        st.stop()

    passages = [p for p, _ in hits]
    scores = [s for _, s in hits]

    if show_ctx:
        st.subheader("📚 Retrieved passages")
        for i, (p, s) in enumerate(zip(passages, scores), 1):
            st.markdown(f"**{i}. (similarity: {s:.4f})**")
            st.write(p)
            st.write("---")

    with st.spinner("Generating answer…"):
        answer = generate_answer(generator, user_q, passages)

    st.subheader("🧠 Answer")
    st.write(answer)
