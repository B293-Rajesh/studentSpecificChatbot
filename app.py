# app.py
import os
import streamlit as st
from index import build_index
from textbook_utils import build_qa_chain, load_pdf_text_or_ocr

DEFAULT_PDF = "x_biology_em.pdf"
GEN_MODEL = "google/flan-t5-base"  # public, small, reliable

st.set_page_config(page_title="Student Specific Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Specific Chatbot")
st.caption("RAG over your textbook — LangChain + FAISS + FLAN-T5 (CPU friendly)")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    pdf_upload = st.file_uploader("Upload PDF textbook (optional)", type=["pdf"])
    top_k = st.slider("Top-K passages", 1, 7, 3)
    show_context = st.checkbox("Show retrieved passages", value=True)

# Save uploaded PDF (if any) to disk
pdf_path = DEFAULT_PDF
if pdf_upload is not None:
    saved = "uploaded.pdf"
    with open(saved, "wb") as f:
        f.write(pdf_upload.read())
    pdf_path = saved

# Build index (cached)
@st.cache_resource(show_spinner="Indexing textbook (this may take a while)...")
def setup_index(pdf_path: str):
    # load raw text (try normal extraction then OCR)
    text = load_pdf_text_or_ocr(pdf_path)
    if not text or text.strip() == "":
        raise RuntimeError("No text could be extracted from the PDF (maybe scanned images). Enable system OCR prerequisites.")
    store = build_index(text)
    return store

# Build QA chain (cached)
@st.cache_resource(show_spinner="Loading generator and QA chain...")
def setup_qa_chain(store):
    qa_chain = build_qa_chain(store, model_id=GEN_MODEL)
    return qa_chain

# Try to build
try:
    store = setup_index(pdf_path)
except Exception as e:
    st.error(f"Indexing failed: {e}")
    st.stop()

try:
    qa_chain = setup_qa_chain(store)
except Exception as e:
    st.error(f"Failed to build QA chain: {e}")
    st.stop()

# Main UI
st.markdown("### Ask a question about the textbook")
question = st.text_input("Your question:")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving relevant passages..."):
        # LangChain retriever lives inside the store (we built it in build_index)
        try:
            # store is a LangChain VectorStore (FAISS) so create a retriever
            retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(question)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

    if not docs:
        st.warning("No relevant passages found.")
        st.stop()

    if show_context:
        st.subheader("📚 Retrieved passages")
        for i, d in enumerate(docs, 1):
            text_preview = d.page_content.strip()
            if len(text_preview) > 800:
                text_preview = text_preview[:800] + "..."
            st.markdown(f"**{i}.** (source: {d.metadata.get('source', 'n/a')})")
            st.write(text_preview)
            st.markdown("---")

    # Run QA chain
    with st.spinner("Generating answer..."):
        try:
            result = qa_chain.run(question, retriever=retriever)
            # Some LangChain chain types return dict; flexible handling:
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("result") or str(result)
            else:
                answer = str(result)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    st.subheader("🧠 Answer")
    st.write(answer)
