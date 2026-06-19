# 📚 Mini RAG System

A lightweight Retrieval-Augmented Generation (RAG) pipeline that ingests a PDF, chunks it intelligently, embeds it with Google's Gemini embedding model, stores/searches it in Pinecone, and now answers questions end-to-end using a tool-calling LangChain agent.


---

## 📁 Folder Structure

```
mini-rag-system/
├── venv/                  # Virtual environment (not tracked in git)
├── .env                   # API keys (Google, Pinecone) — not tracked in git
├── .gitignore
├── fullstack_guide.pdf     # Source PDF to be ingested
├── main.py                 # Core script: ingestion + RAG agent
└── README.md
```

## ✨ What it does

1. **Loads** a PDF (`fullstack_guide.pdf`) using `PyPDFLoader`
2. **Cleans** the extracted text
3. **Splits** it into overlapping chunks using `RecursiveCharacterTextSplitter`
4. **Embeds** each chunk with Google's `gemini-embedding-001` model
5. **Stores** the embeddings in a Pinecone vector index
6. **Searches** the index semantically (not just keyword matching!)
7. **Retrieves + Generates**: a LangChain agent calls a custom `getcontext` tool to pull relevant chunks from Pinecone, then uses Gemini 2.5 Flash to generate a grounded answer

---

## 🛠 Tech Stack

![LangChain](https://img.shields.io/badge/LangChain-000?style=for-the-badge&logo=chainlink&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-00BFFF?style=for-the-badge)
![PyPDFLoader](https://img.shields.io/badge/PyPDFLoader-FF4B4B?style=for-the-badge)
![Text Splitter](https://img.shields.io/badge/Text%20Splitter-8A2BE2?style=for-the-badge)
![LangChain Agents](https://img.shields.io/badge/LangChain%20Agents-1C3C3C?style=for-the-badge)
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

## 🔍 How retrieval works (low-level)

Under the hood, similarity search against Pinecone looks like this:
```python
vector_store.similarity_search(query="version control with git", k=1)
```
This returns the most relevant chunk(s) from your PDF based on **meaning**, not exact keyword matches.

---

## 🤖 RAG Agent (query → retrieve → generate)

Instead of calling `similarity_search` manually, the project now wraps retrieval in a **tool** and hands it to a LangChain agent powered by **Gemini 2.5 Flash**:

```python
@tool
def getcontext(query: str):
    """Use this tool to get more information to fulfill the user's request."""
    result = vector_store.similarity_search(query=query, k=2)
    return str(result)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

agent = create_agent(
    model=model,
    tools=[getcontext],
    system_prompt=(
        "You must ALWAYS call the getcontext tool first to retrieve "
        "relevant information before answering any question. "
        "Base your answer only on the retrieved context."
    )
)

response = agent.invoke({
    "messages": [HumanMessage("What is Fullstack development?")]
})
```

**Why a tool instead of a direct call?**
- The agent decides *when* to retrieve, rather than retrieval being hardcoded into every query.
- The `system_prompt` enforces a **tool-first workflow**, so the model is grounded in your Pinecone data instead of answering from its own training knowledge (reduces hallucination).
- This sets up the project for **multi-tool agents** later (web search, calculators, etc.) without changing the core retrieval logic.

---

## ⚠️ Known Quirks (a.k.a. things I learned the hard way)

- `index.delete(delete_all=True)` throws a `NotFoundException` if the namespace is empty/new — wrap it in a `try/except` if running on a fresh index.
- Vector writes to Pinecone aren't instantly searchable — add a short `time.sleep()` after `add_documents()` if you're querying right away.
- Chunk boundaries aren't perfect with character-based splitting — a chunk might blend two topics together if they're short and close together. This is normal; retrieval + LLM generation downstream usually compensates for it.
- `create_agent()` expects `system_prompt`, not `prompt` — passing `prompt=` raises a `TypeError`.
- The agent only calls `getcontext` if it judges the question needs it. For very generic questions, it may answer from its own knowledge unless the system prompt explicitly forces tool-first behavior (as done above).

