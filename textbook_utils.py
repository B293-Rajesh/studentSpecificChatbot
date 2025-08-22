import PyPDF2
from sentence_transformers import SentenceTransformer
from index import SimpleVectorStore

def load_textbook(file_path):
    """Load PDF and extract text"""
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def index_textbook(chunks):
    """Embed chunks and store in vector DB"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    store = SimpleVectorStore()

    for idx, chunk in enumerate(chunks):
        embedding = model.encode([chunk])[0]
        store.add(embedding, idx)

    return model, store
