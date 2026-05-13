# ChatWithDocs
# 📓 NotebookLM Clone — RAG-Powered Document Q&A

A local NotebookLM alternative built with LangChain, Groq, and Streamlit.  
Upload PDFs, DOCX, or TXT files and chat with your documents using a powerful 
RAG pipeline with semantic chunking, streaming responses, and multi-document comparison.

![Demo](assets/demo.gif)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-format upload | Supports PDF, DOCX, and TXT files |
| 🔍 Semantic chunking | Groups sentences by meaning, not fixed character count |
| 💬 Streaming responses | Answers stream word-by-word like ChatGPT |
| 🧠 Context memory | Retrieved chunks saved across turns so context isn't lost |
| 📚 Multi-doc comparison | Ask "what does doc A vs doc B say about X" |
| 🔦 Source highlighting | Exact matched sentence shown for each answer |
| 📥 Export chat | Download full conversation as .txt or .md |
| ♻️ Smart re-indexing | Hash check prevents re-processing same files |

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit
- **LLM** — Llama 3.3 70B via [Groq](https://groq.com)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store** — ChromaDB (in-memory)
- **RAG Framework** — LangChain
- **Chunking** — SemanticChunker (LangChain Experimental)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/notebooklm.git
cd notebooklm
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
# Copy the example env file
cp .env.example .env

# Open .env and add your Groq API key
# Get a free key at: https://console.groq.com
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
notebooklm/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── .gitignore           # Files excluded from git

## 🙋 Author

Built by [Your Name](https://github.com/YOUR_USERNAME)
