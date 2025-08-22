from sentence_transformers import SentenceTransformer
from index import SimpleVectorStore

def load_textbook(file_path="textbook.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def index_textbook(chunks):
    # ⚡ FIX: no device="cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    store = SimpleVectorStore(dim=embeddings.shape[1])

    for i, emb in enumerate(embeddings):
        store.add(chunks[i], emb)

    return model, store
