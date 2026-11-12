# 🎓 Campus Companion – Interactive RAG Learning Lab

**Campus Companion** is a beginner-friendly, transparent **Retrieval-Augmented Generation (RAG)** application built to teach *how* RAG works internally — not just provide a chatbot. Every stage of document ingestion and question answering is visible and inspectable.

> **Fictional Policy Notice:** All included sample documents belong to *Horizon Crest University (Fictional)* and are strictly for educational demonstration.

---

## 1. RAG Architecture

```
Documents (PDF/TXT)
   │
   ▼  1. Text Extraction (PyMuPDF / UTF-8)
Cleaned Text
   │
   ▼  2. Chunking (LangChain RecursiveCharacterTextSplitter: 500 chars, 100 overlap)
Text Chunks
   │
   ▼  3. Embeddings (Sentence Transformers: all-MiniLM-L6-v2, 384 dims)
Dense Vectors + Metadata
   │
   ▼  4. Vector Database (ChromaDB persistent local storage)
Indexed Knowledge Base
   │
   ▼  5. Retrieval (Embed User Question -> Cosine Similarity Top-K Search)
Top-K Relevant Chunks
   │
   ▼  6. Prompt Augmentation (System Instructions + Retrieved Context + User Question)
Augmented Prompt
   │
   ▼  7. LLM Generation (Groq API)
Grounded Answer + Source Citations
```

---

## 2. Technology Stack

| Technology | Role in This Project |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Streamlit** | Dark-mode SaaS dashboard, chat UI, and visual pipeline explorer |
| **PyMuPDF (`fitz`)** | Extracts text page-by-page from uploaded PDF files |
| **LangChain Text Splitters** | Splits documents into overlapping chunks (`RecursiveCharacterTextSplitter`) |
| **Sentence Transformers (`all-MiniLM-L6-v2`)** | Generates 384-dimensional local embeddings without an external API |
| **ChromaDB** | Stores vectors, text, and metadata persistently in `data/chroma_db/` |
| **Groq API** | Fast LLM inference for grounded answer generation |
| **python-dotenv** | Loads `GROQ_API_KEY` safely from `.env` |

---

## 3. Windows Setup & Installation (VS Code)

### Prerequisites
- Python 3.10 – 3.12 installed on Windows
- VS Code (optional, recommended)
- A free Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys)

### Step 1: Open Terminal in `campus_companion`
```powershell
cd campus_companion
```

### Step 2: Create & Activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Your Groq API Key
1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` and replace `your_groq_api_key_here` with your real Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

### Step 5: Run the Application
```powershell
streamlit run app.py
```
Your browser will open automatically (usually at `http://localhost:8501`).

---

## 4. How a Question Flows Through the Code

1. **UI Input (`components/chat.py`)**: You type a question in Streamlit.
2. **Pipeline Orchestrator (`rag/pipeline.py`)**: Calls `run_rag_query(...)`.
3. **Query Embedding (`rag/embeddings.py`)**: Converts your question into a 384-D vector using `all-MiniLM-L6-v2`.
4. **Vector Search (`rag/vector_store.py` & `rag/retrieval.py`)**: Searches ChromaDB using cosine similarity and returns the Top-K chunks.
5. **Prompt Augmentation (`rag/prompt_builder.py`)**: Combines System Instructions + Retrieved Context + Your Question.
6. **LLM Generation (`rag/generation.py`)**: Sends the augmented prompt to Groq (or reports a clear missing-key message if `GROQ_API_KEY` is not set).
7. **Visual Inspection (`components/pipeline_explorer.py`)**: Click **Explore How This Answer Was Generated** to inspect all 7 stages!

---

## 5. Project Structure

```
campus_companion/
├── app.py                     # Main Streamlit entry point & dark SaaS theme
├── config.py                  # Paths, default RAG settings, .env loader
├── requirements.txt           # Python dependencies
├── .env.example               # Template for GROQ_API_KEY
├── components/
│   ├── sidebar.py             # Navigation & live DB/LLM status
│   ├── dashboard.py           # KPIs, clickable 8-stage diagram, activity log
│   ├── knowledge_base.py      # Upload, 5-step progress, chunk inspector, delete
│   ├── chat.py                # Conversational RAG chatbot
│   ├── pipeline_explorer.py   # 7-stage visual RAG inspector
│   ├── experiments.py         # Side-by-side Top-K & chunking experiments
│   ├── learning_centre.py     # 13 RAG concepts + searchable glossary
│   └── settings_about.py      # Configurable defaults & session key
├── rag/
│   ├── document_loader.py     # PyMuPDF & TXT extraction
│   ├── text_cleaner.py        # Text normalization
│   ├── chunking.py            # RecursiveCharacterTextSplitter
│   ├── embeddings.py          # all-MiniLM-L6-v2 local embeddings
│   ├── vector_store.py        # ChromaDB persistent CRUD & query
│   ├── retrieval.py           # Top-K, threshold, sentence highlighting
│   ├── prompt_builder.py      # Structured prompt augmentation
│   ├── generation.py          # Groq API integration
│   └── pipeline.py            # Ingestion & 7-stage query orchestration
├── sample_documents/          # 5 fictional Horizon Crest University policies
├── data/                      # Persistent ChromaDB, uploads, registry JSON
└── utils/
    └── helpers.py             # Hashing, formatting, registry & activity log
```

---

## 6. Common Errors & Solutions

- **Missing Groq API Key warning:** Create `.env` from `.env.example` and paste your `GROQ_API_KEY`, or enter it temporarily in **Settings & About**.
- **Empty PDF error:** The PDF is likely a scanned image without selectable text. Try a text-based PDF or `.txt` file.
- **Duplicate document warning:** Each file is hashed with SHA-256. Use **Reprocess** or **Delete** in **Knowledge Base** to re-index it.
- **First-run model download:** On the very first run, `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) once and caches it locally.

