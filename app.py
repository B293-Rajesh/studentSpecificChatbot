# app.py
import streamlit as st
from textbook_utils import load_textbook_pdf, chunk_text, index_textbook, query_textbook

st.set_page_config(page_title="Student Specific Chatbot", page_icon="🤖", layout="wide")

st.title("🎓 Student Specific Chatbot")
st.write("Welcome! Ask me a question about your syllabus or notes.")

# Load and index the PDF once
filepath = "x_biology_em.pdf"  # 👈 change to your actual file
text = load_textbook_pdf(filepath)
chunks = chunk_text(text)
model, store = index_textbook(chunks)

# Chatbot UI
user_input = st.text_input("Your question:")

if user_input:
    st.write(f"**You asked:** {user_input}")

    results = query_textbook(model, store, user_input, top_k=3)

    st.subheader("📖 Top 3 Relevant Passages")
    for i, (chunk, dist) in enumerate(results, 1):
        st.markdown(f"**{i}. (similarity score: {dist:.4f})**")
        st.write(chunk)
