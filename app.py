import streamlit as st
from textbook_utils import build_qa_chain

st.set_page_config(page_title="📚 Student Specific Chatbot", page_icon="🤖", layout="wide")

st.title("🎓 Student Specific Chatbot")
st.write("Upload your textbook (PDF) and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        qa = build_qa_chain("uploaded.pdf")
        st.success("✅ PDF successfully indexed with OCR!")
        query = st.text_input("Ask a question about your textbook:")
        if query:
            result = qa.run(query)
            st.write("**Answer:**", result)
    except Exception as e:
        st.error(f"❌ Indexing failed: {e}")
else:
    st.info("Please upload a PDF to get started.")
