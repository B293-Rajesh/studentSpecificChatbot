import streamlit as st
from index import load_index
from textbook_utils import query_index
from transformers import pipeline

@st.cache_resource
def setup():
    store, chunks = load_index("x_biology_em.pdf")
    # Simple text generator
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return store, generator

st.title("🎓 Student Specific Chatbot")
st.write("RAG over your textbook → simple generator (FLAN-T5) for answers")

store, generator = setup()

user_input = st.text_input("Your question:")

if user_input:
    st.write(f"**You asked:** {user_input}")

    # Retrieve top 3 chunks
    results = query_index(store, user_input, top_k=3)

    st.subheader("📚 Retrieved passages")
    for r, score in results:
        st.write(f"- (similarity: {score:.4f}) {r}")

    # Prepare context for LLM
    context = " ".join([r for r, _ in results])
    prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {user_input}\n\nAnswer in one short sentence."

    output = generator(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]

    st.subheader("🧠 Answer")
    st.write(output)
