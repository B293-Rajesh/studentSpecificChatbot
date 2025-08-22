# index.py
from typing import Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_index(full_text: str) -> Any:
    """
    Build a LangChain FAISS vector store from raw full_text.
    Returns the FAISS store (LangChain object) which implements .as_retriever().
    """
    # Split full_text into Document objects with metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    docs = splitter.create_documents([full_text])

    # Embeddings: explicit CPU device to avoid device errors
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Build FAISS index via LangChain community wrapper
    store = FAISS.from_documents(docs, embedder)
    return store
