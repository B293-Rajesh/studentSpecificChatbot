import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# ----------- PDF Loader using PyMuPDF -----------
def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# ----------- Index Builder -----------
def build_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_texts(chunks, embeddings)
    return store

# ----------- QA Chain -----------
def build_qa(store):
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # lightweight open model
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ----------- Streamlit UI -----------
st.title("🎓 Student Specific Chatbot")
st.write("Upload your textbook (PDF) and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Extracting and indexing PDF..."):
        text = load_pdf_text("uploaded.pdf")
        if not text.strip():
            st.error("❌ No text found in PDF. This may be a scanned (image-only) file.")
        else:
            store = build_index(text)
            qa = build_qa(store)
            st.success("✅ PDF indexed! Ask your questions below:")

            query = st.text_input("Ask a question about your textbook:")
            if query:
                with st.spinner("Generating answer..."):
                    answer = qa.run(query)
                    st.write("**Answer:**", answer)
