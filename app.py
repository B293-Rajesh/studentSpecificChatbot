import streamlit as st
import test as t
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
text = t.load_textbook_pdf(filepath)
chunks = t.chunk_text(text)
model, store = t.index_textbook(chunks)
st.write(t.query_textbook(model, store, user_input))
