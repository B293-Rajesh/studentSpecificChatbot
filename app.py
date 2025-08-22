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
    model, store, chunks = index_pdf(pdf_file)

    # Use public Flan-T5 model for text generation
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return model, store, chunks, tokenizer, llm

if uploaded_file:
    model, store, chunks, tokenizer, llm = setup(uploaded_file)

    if user_input:
        # Get query embedding
        query_vec = model.encode([user_input])[0]
        results = store.similarity_search(user_input, k=3)
        st.write("📚 Top 3 Relevant Passages")
        context = ""
        for i, doc in enumerate(results, 1):
            st.write(f"{i}. {doc.page_content[:300]}...")
            context += doc.page_content + "\n"

        # Generate response
        prompt = f"Use the following context to answer the question:\n{context}\nQuestion: {user_input}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = llm.generate(**inputs, max_new_tokens=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("🧠 Answer")
        st.write(answer)
