import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    # Load LLaMA model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    llm = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, store, chunks, tokenizer, llm

if uploaded_file:
    pdf_path = uploaded_file
    model, store, chunks, tokenizer, llm = setup(pdf_path)

    if user_input:
        # Get query embedding
        query_vec = model.encode([user_input])[0]
        results = store.search(query_vec, k=3)
        st.write("📚 Top 3 Relevant Passages")
        context = ""
        for i, (chunk, dist) in enumerate(results, 1):
            st.write(f"{i}. (similarity: {dist:.4f}) {chunk[:300]}...")
            context += chunk + "\n"

        # Generate response
        prompt = f"Use the following context to answer the question:\n{context}\nQuestion: {user_input}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        outputs = llm.generate(**inputs, max_new_tokens=150)
        answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        st.write("🧠 Answer")
        st.write(answer)
