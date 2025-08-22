import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textbook_utils import index_pdf
import torch

st.set_page_config(page_title="Student Specific Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Specific Chatbot")
st.write("Upload your textbook (PDF) and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
user_input = st.text_input("Your question:")

@st.cache_data
def setup(pdf_file):
    store = index_pdf(pdf_file)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return store, tokenizer, llm

if uploaded_file:
    store, tokenizer, llm = setup(uploaded_file)

    if user_input:
        # Retrieve relevant chunks
        results = store.similarity_search(user_input, k=3)
        context = " ".join([doc.page_content for doc in results])

        # Generate answer using Flan-T5
        prompt = f"Answer the question based on context:\n{context}\nQuestion: {user_input}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = llm.generate(**inputs, max_new_tokens=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.write("🧠 Answer")
        st.write(answer)
