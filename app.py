import streamlit as st
from textbook_utils import load_textbook, chunk_text, index_textbook
from index import SimpleVectorStore
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline

# Load Hugging Face LLaMA model
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

# Setup embeddings, vector store, and chunks
@st.cache_resource
def setup():
    text = load_textbook("x_biology_em.pdf")   # your syllabus PDF
    chunks = chunk_text(text)
    model, store = index_textbook(chunks)
    llm = load_llm()
    return model, store, llm, chunks

def answer_question(question, model, store, llm, chunks):
    query_emb = model.encode([question])[0]
    top_chunks = store.search(query_emb, top_k=3)

    # Build context from retrieved chunks
    context = "\n\n".join([chunks[i] for i in top_chunks])
    prompt = f"""You are a helpful biology tutor. 
Answer the question based only on the following textbook context. 
If not enough info is present, say you are not sure.

Context:
{context}

Question: {question}
Answer:"""

    response = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# Streamlit UI
st.title("🎓 Student Specific Chatbot")
st.write("Welcome! Ask me a question about your syllabus or notes.")

model, store, llm, chunks = setup()

user_input = st.text_input("Your question:")

if user_input:
    st.write(f"You asked: {user_input}")
    answer = answer_question(user_input, model, store, llm, chunks)
    st.write("📝 Answer:")
    st.write(answer)
