import streamlit as st
from index import load_index
from textbook_utils import build_qa_chain

@st.cache_resource
def setup():
    store = load_index("x_biology_em.pdf")
    qa = build_qa_chain(store)
    return qa

st.title("🎓 Student Specific Chatbot")
st.write("RAG over your textbook → using LangChain + FLAN-T5")

qa = setup()

user_input = st.text_input("Your question:")

if user_input:
    st.write(f"**You asked:** {user_input}")

    result = qa({"query": user_input})

    # Show retrieved passages
    st.subheader("📚 Retrieved passages")
    for doc in result["source_documents"]:
        st.write(f"- {doc.page_content[:200]}...")

    # Show answer
    st.subheader("🧠 Answer")
    st.write(result["result"])
