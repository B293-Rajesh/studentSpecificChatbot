import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using OCR (for scanned/image-only PDFs)."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
    return text


def load_pdf_text(pdf_path):
    """Load text from PDF. Falls back to OCR if needed."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

    # First try normal text extraction
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # If nothing extracted → run OCR
    if not docs or all(d.page_content.strip() == "" for d in docs):
        extracted_text = extract_text_with_ocr(pdf_path)
        if not extracted_text.strip():
            raise RuntimeError("OCR failed: Could not extract any text from PDF.")
        from langchain.schema import Document
        docs = [Document(page_content=extracted_text)]
    return docs


def build_qa_chain(pdf_path):
    docs = load_pdf_text(pdf_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    store = FAISS.from_documents(chunks, embedder)

    retriever = store.as_retriever(search_kwargs={"k": 3})

    # HuggingFace model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    # Prompt template
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

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa
