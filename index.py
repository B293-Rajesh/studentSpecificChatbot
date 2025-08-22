from textbook_utils import load_textbook_pdf, chunk_text, index_textbook, retrieve_top_k

if __name__ == "__main__":
    text = load_textbook_pdf("x_biologyA_em.pdf")
    chunks = chunk_text(text, chunk_size=500, overlap=60)
    emb_model, store = index_textbook(chunks)
    print(f"Indexed {len(chunks)} chunks.")
    hits = retrieve_top_k(emb_model, store, "Explain Kwashiorkor disease", k=3)
    for i, (p, s) in enumerate(hits, 1):
        print(f"\n{i}. score={s:.4f}\n{p[:400]}...")
