import streamlit as st
from textbook_utils import load_textbook, chunk_text, index_textbook

st.title("🎓 Student Specific Chatbot")
st.write("Welcome! Ask me a question about your syllabus or notes.")

# Load and index textbook once
@st.cache_resource
def setup():
    text = load_textbook("textbook.txt")
    chunks = chunk_text(text)
    return index_textbook(chunks)

model, store = setup()

# User input
user_input = st.text_input("Your question:")

if user_input:
    st.write(f"You asked: {user_input}")

    # Encode query
    query_embedding = model.encode(user_input, convert_to_numpy=True)

    # Search top 3 chunks
    results = store.search(query_embedding, top_k=3)

    st.subheader("📖 Top 3 Relevant Passages")
    for i, (text, score) in enumerate(results, 1):
        st.write(f"**{i}. (similarity score: {score:.4f})**")
        st.write(text)
        st.write("---")
