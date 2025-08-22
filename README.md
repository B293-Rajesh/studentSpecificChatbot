# 🎓 Student Specific Chatbot

This app allows students to upload their textbooks (PDFs) and ask questions about them using **LangChain + HuggingFace + OCR**.  

It works with:
- **Text-based PDFs** (normal, copyable text)  
- **Image-only PDFs** (scanned pages) → handled with OCR (Tesseract)  

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/studentspecificchatbot.git
cd studentspecificchatbot
```

### 2. Install system dependencies
This project requires **Tesseract OCR** and **Poppler**.

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

#### On macOS (with Homebrew):
```bash
brew install tesseract poppler
```

#### On Windows:
1. Download and install **Tesseract OCR** from:  
   👉 https://github.com/UB-Mannheim/tesseract/wiki  
   - After installation, add Tesseract’s install path (e.g., `C:\Program Files\Tesseract-OCR`) to your **System PATH**.  

2. Download and install **Poppler for Windows** from:  
   👉 https://github.com/oschwartz10612/poppler-windows/releases/  
   - Extract the ZIP file and add the `bin/` folder to your **System PATH**.  

3. Restart your terminal/IDE so changes take effect.  

---

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
