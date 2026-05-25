# 🤖 RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with LangChain, Mistral AI, and ChromaDB. Upload a PDF and chat with your documents intelligently — deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42.2-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo

> Try the app live — no setup required!

👉 **[Open RAG Chatbot on Streamlit Cloud](https://ragchatbot-ztpmagxgynuskca5nht3kb.streamlit.app/)**

---

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF and ask questions about its contents
- 🧠 **Mistral AI** — Powered by `mistral-large-latest` for accurate, context-aware answers
- 🗃️ **ChromaDB Vector Store** — Fast semantic search over your document chunks
- 🔗 **LangChain RAG Pipeline** — Chunking → Embedding → Retrieval → Generation
- 🖥️ **Streamlit UI** — Clean, interactive chat interface
- ☁️ **Streamlit Cloud Deployment** — One-click deploy, no server needed

---

## 🏗️ Architecture

```
User uploads PDF
      │
      ▼
PyPDFLoader → RecursiveCharacterTextSplitter
      │
      ▼
HuggingFace Embeddings (sentence-transformers)
      │
      ▼
ChromaDB Vector Store
      │
      ▼
User asks question → Retriever fetches top-k chunks
      │
      ▼
Mistral AI LLM → Answer
      │
      ▼
Streamlit Chat UI
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ragchatbot.git
cd ragchatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your Mistral API key at [console.mistral.ai](https://console.mistral.ai).

### 5. Run the app

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud

### Step 1 — Push your code to GitHub

Make sure your repo contains:
```
ragchatbot/
├── main.py
├── requirements.txt
├── runtime.txt          ← contains: python-3.11
├── .env                 ← DO NOT commit this (add to .gitignore)
└── README.md
```

### Step 2 — Add `.env` to `.gitignore`

```bash
echo ".env" >> .gitignore
```

### Step 3 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** to `main.py`
5. Click **"Advanced settings"** → **Secrets** and add:

```toml
MISTRAL_API_KEY = "your_mistral_api_key_here"
```

6. Click **"Deploy"**

### Step 4 — Add `runtime.txt`

Create a file named `runtime.txt` in your repo root with:
```
python-3.11
```

This tells Streamlit Cloud to use Python 3.11 (required for compatibility).

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Mistral AI (`mistral-large-latest`) |
| **Embeddings** | HuggingFace `sentence-transformers` |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain 0.3.x |
| **Document Loader** | PyPDFLoader |
| **Text Splitter** | RecursiveCharacterTextSplitter |
| **UI** | Streamlit |
| **Language** | Python 3.11 |

---

## 📁 Project Structure

```
ragchatbot/
├── main.py              # Streamlit app entry point
├── requirements.txt     # Python dependencies (pinned versions)
├── runtime.txt          # Python version for Streamlit Cloud
├── .env                 # API keys (local only, not committed)
├── .gitignore
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `MISTRAL_API_KEY` | Your Mistral AI API key | ✅ Yes |

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError: No module named 'langchain_huggingface'`**
Add `langchain-huggingface` to `requirements.txt`. LangChain integrations are separate packages.

**`ImportError` on Streamlit Cloud**
Make sure `runtime.txt` contains `python-3.11`. Python 3.14 is not yet fully supported by LangChain dependencies.

**App crashes on large PDFs**
Streamlit Cloud has memory limits. For large documents, reduce chunk size in `RecursiveCharacterTextSplitter`.

**`google-generativeai` import error**
This package reached end-of-life on Nov 30, 2025. Replace it with `google-genai` in your requirements.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- [LangChain](https://langchain.com) — RAG framework
- [Mistral AI](https://mistral.ai) — LLM provider
- [ChromaDB](https://trychroma.com) — Vector database
- [Streamlit](https://streamlit.io) — UI & deployment platform
- [HuggingFace](https://huggingface.co) — Embeddings models
