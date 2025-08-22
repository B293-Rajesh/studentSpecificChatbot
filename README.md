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
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
brew install tesseract poppler
pip install -r requirements.txt
streamlit run app.py
