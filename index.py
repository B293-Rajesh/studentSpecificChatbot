from textbook_utils import load_textbook, chunk_text, index_textbook

if __name__ == "__main__":
    # Load your PDF
    text = load_textbook("x_biology_em.pdf")
    chunks = chunk_text(text)
    model, store = index_textbook(chunks)

    print(f"✅ Indexed {len(chunks)} chunks from the textbook.")
