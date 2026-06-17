# 📚 Mini RAG System

A lightweight Retrieval-Augmented Generation (RAG) pipeline that ingests a PDF, chunks it intelligently, embeds it with Google's Gemini embedding model, and stores/searches it in Pinecone.

> Built as a learning project — more tools and features coming soon 🚀

---

## ✨ What it does

1. **Loads** a PDF (`fullstack_guide.pdf`) using `PyPDFLoader`
2. **Cleans** the extracted text
3. **Splits** it into overlapping chunks using `RecursiveCharacterTextSplitter`
4. **Embeds** each chunk with Google's `gemini-embedding-001` model
5. **Stores** the embeddings in a Pinecone vector index
6. **Searches** the index semantically (not just keyword matching!)

---
## 🛠 Tech Stack

![LangChain](https://img.shields.io/badge/LangChain-000?style=for-the-badge&logo=chainlink&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-00BFFF?style=for-the-badge)
![PyPDFLoader](https://img.shields.io/badge/PyPDFLoader-FF4B4B?style=for-the-badge)
![Text Splitter](https://img.shields.io/badge/Text%20Splitter-8A2BE2?style=for-the-badge)
![dotenv](https://img.shields.io/badge/dotenv-ECD53F?style=for-the-badge&logo=python&logoColor=black)


---

## 📦 Setup

### 1. Clone & install dependencies
```bash
pip install langchain langchain-google-genai langchain-pinecone langchain-community langchain-text-splitters pinecone python-dotenv pypdf
```

### 2. Create a `.env` file
```env
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Create your Pinecone index
Make sure you have an index named `rag-index1` with:
- **Dimension:** `3072` (matches `gemini-embedding-001` output)
- **Metric:** `cosine`

### 4. Add your PDF
Drop your PDF file in the project root and update the filename in the script:
```python
loader = PyPDFLoader("./fullstack_guide.pdf")
```

### 5. Run it
```bash
python main.py
```

---

## 🔍 How the search works

The script ends by running a semantic similarity search:
```python
vector_store.similarity_search(query="version control with git", k=1)
```
This returns the most relevant chunk(s) from your PDF based on **meaning**, not exact keyword matches.

---

## ⚠️ Known Quirks (a.k.a. things I learned the hard way)

- `index.delete(delete_all=True)` throws a `NotFoundException` if the namespace is empty/new — wrap it in a `try/except` if running on a fresh index.
- Vector writes to Pinecone aren't instantly searchable — add a short `time.sleep()` after `add_documents()` if you're querying right away.
- Chunk boundaries aren't perfect with character-based splitting — a chunk might blend two topics together if they're short and close together. This is normal; retrieval + LLM generation downstream usually compensates for it.

---

## 🗺️ Roadmap

- [ ] Add more tools (web search, calculator, etc.)
- [ ] Hook up an LLM for full RAG (retrieve + generate answers)
- [ ] Support multiple PDFs / batch ingestion
- [ ] Add a simple CLI or UI for querying

