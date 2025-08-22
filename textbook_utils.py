from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import os

def load_pdf_text(pdf_path):
    """Try to load text from a PDF without OCR fallback."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs or all(d.page_content.strip() == "" for d in docs):
            raise ValueError("No text detected in PDF. This file may be scanned (image-only).")
        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF text: {e}")

def build_qa_chain(pdf_path):
    # Load text from PDF
    docs = load_pdf_text(pdf_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store vectors
    store = FAISS.from_documents(chunks, embedder)

    # Define retriever
    retriever = store.as_retriever(search_kwargs={"k": 3})

    # HuggingFace model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    # Prompt
    prompt_template = """You are a helpful tutor. 
Answer the question based on the provided context.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Retrieval-based QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa
