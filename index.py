# index.py
from textbook_utils import load_textbook_pdf, chunk_text, index_textbook

if __name__ == "__main__":
    text = load_textbook_pdf("x_biology_em.pdf")  # 👈 change if needed
    chunks = chunk_text(text)
    model, store = index_textbook(chunks)
    print(f"✅ Indexed {len(chunks)} chunks successfully!")
