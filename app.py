import streamlit as st
from myvectorstore import index
import test
st.set_page_config(page_title="Student Specific Chatbot", page_icon="🤖", layout="wide")

st.title("🎓 Student Specific Chatbot")
st.write("Welcome! Ask me a question about your syllabus or notes.")

# Example placeholder input/output
user_input = st.text_input("Your question:")

if user_input:
    # For now, just echo back the input
    st.write(f"**You asked:** {user_input}")
    st.write("*(This is where the chatbot's answer will appear.)*")

filepath = "x_biology_em.pdf"  # Change this
question = "What is respiration?"

text = load_textbook_pdf(filepath)
chunks = chunk_text(text)
model, store = index_textbook(chunks)
query_textbook(model, store, question)
