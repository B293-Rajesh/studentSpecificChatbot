import streamlit as st
from textbook_utils import load_textbook, chunk_text, index_textbook
from sentence_transformers import SentenceTransformer

@st.cache_resource
def setup():
    text = load_textbook("x_biology_em.pdf")  # load from your PDF
    chunks = chunk_text(text)
    return index_textbook(chunks)

model, store = setup()

st.title("🎓 Student Specific Chatbot")
st.write("Welcome! Ask me a question about your syllabus or notes.")

user_input = st.text_input("Your question:")

if user_input:
    query_vec = model.encode(user_input)
    results = store.query(query_vec, top_k=3)

    st.write(f"You asked: {user_input}")
    st.write("📖 Top 3 Relevant Passages")

    for i, passage in enumerate(results, 1):
        st.write(f"{i}. {passage}")
