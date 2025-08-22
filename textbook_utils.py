import fitz  # PyMuPDF

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            # Split into smaller chunks
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text_chunks.extend(lines)
    return text_chunks
