# 🇻🇳 Vietnamese Road Traffic Law Chatbot (RAG-based)

A legal question-answering system for Vietnamese road traffic laws using Retrieval-Augmented Generation (RAG), FAISS retriever, reranker, and context-based answer generation.

---

## 📌 Features

- ✅ Retrieve relevant legal paragraphs from a law corpus using FAISS
- ✅ Rerank retrieved contexts using cross-encoder reranker
- ✅ Generate accurate and concise answers from retrieved legal content
- ✅ Modular architecture: retriever, reranker, generator
- ✅ Easily extensible with custom corpus

---

## 🏗️ Architecture Overview

```text
User Question
      ↓
 Retriever (bi-encoder → FAISS search)
      ↓
 Reranker (cross-encoder)
      ↓
 Answer Generation (contextual response)
      ↓
 Answer
```

---

## 📁 Project Structure

```text
.
├── main.py                # Entry point for chat interaction
├── retriever.py           # FAISS + embedding-based retriever
├── reranker.py            # Cross-encoder reranker module
├── generator.py           # RAG pipeline (retriever → reranker → generator)
├── data/                  # Folder with corpus, index, etc.
│   └── corpus.csv
├── faiss/                 # Saved FAISS index files
├── .env                   # Environment variables
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/NobtH/Vietnamese-road-traffic-QA-chatbot-RAG.git
cd Vietnamese-road-traffic-QA-chatbot-RAG
```

2. (Recommended) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the root directory:

```env
API_KEY=your_api_key
API_BASE=https://api.service.com/v1
```

---

## 🚀 Run the Chatbot

Run the chatbot CLI:

```bash
python main.py
```
 unicorn main:app --reload 

Example interaction:

```text
❓ Mức phạt khi không đội mũ bảo hiểm là bao nhiêu?
📝 [Câu trả lời sinh từ luật + context]
```

---

## 🧠 Model Used

- **Retriever**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 📚 Data Format

Corpus format (`corpus.csv`):

```csv
cid,law_number,chapter,article,text
1,36/2024/QH15,I,6,"Người điều khiển xe mô tô không đội mũ bảo hiểm..."
```

---

## 📜 License

MIT License. See [LICENSE](./LICENSE) for details.

---

## 🙋‍♂️ Contact

- Author: [NobtH](https://github.com/NobtH)