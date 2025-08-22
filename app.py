import streamlit as st
from index import load_index
from textbook_utils import query_index
from transformers import pipeline

# ----------------------------
# Load model with safer decoding
# ----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base",   # You can also swap with "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer="google/flan-t5-base",
        device_map="auto"
    )

# ----------------------------
# Setup everything
# ----------------------------
@st.cache_resource
def setup():
    store, chunks = load_index("x_biologyA_em.pdf")
    llm = load_llm()
    return store, chunks, llm

# ----------------------------
# Helper: clean duplicates
# ----------------------------
def clean_answer(text):
    words = text.replace(",", "").split()
    seen = set()
    result = []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            result.append(w)
    return " ".join(result)

# ----------------------------
# Main App
# ----------------------------
st.title("🎓 Student Specific Chatbot")
st.write("Welcome! Ask me a question about your syllabus or notes.")

user_input = st.text_input("Your question:")

if user_input:
    store, chunks, llm = setup()

    # Get top 3 passages
    retrieved_texts = query_index(store, user_input, top_k=3)
    st.subheader("📚 Retrieved passages")
    for i, passage in enumerate(retrieved_texts):
        st.write(f"{i+1}. {passage}")

    # Prompt construction
    context = "\n".join(retrieved_texts)
    prompt = f"""
    You are a helpful teacher. Use the following textbook passages to answer.

    Passages:
    {context}

    Question:
    {user_input}

    Answer clearly in 1-2 sentences. 
    If the question asks for names, list them without repeating.
    """

    # Generate answer with better decoding
    response = llm(
        prompt,
        max_new_tokens=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    raw_answer = response[0]['generated_text']
    final_answer = clean_answer(raw_answer)

    st.subheader("🧠 Answer")
    st.write(final_answer)
