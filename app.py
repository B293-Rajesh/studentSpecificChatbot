import streamlit as st
from index import load_index

@st.cache_resource
def setup(pdf_path):
    return load_index(pdf_path)

st.title("📘 RAG over Your Textbook")

pdf_file = st.file_uploader("Upload your textbook (PDF)", type=["pdf"])

if pdf_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())

    qa = setup("uploaded.pdf")
    st.success("✅ Textbook indexed successfully!")
