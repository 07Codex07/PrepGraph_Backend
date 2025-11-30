# PrepGraph — Hybrid RAG Chat System (FastAPI + Groq + FAISS)

PrepGraph is a backend-only Retrieval-Augmented Generation (RAG) system that provides a production-ready chat API. It uses FastAPI, Groq Llama-3.1, FAISS vector search, BM25 keyword search, SentenceTransformer embeddings, and SQLite-based long-term memory. It is designed to be used with any frontend such as React, Next.js, or Vue.

The system automatically downloads academic PDFs (syllabus + PYQs) from HuggingFace, indexes them using semantic + keyword search, retrieves the most relevant content for each query, and passes the cleaned context along with the last user messages to the LLM to generate an accurate response. It is fully dockerized and deployable.

---

## Components

### 1. memory_store.py — Persistent Memory
- Stores all messages (user + assistant) in SQLite.
- Auto-prunes old messages (MAX_MESSAGES_PER_USER = 500).
- Functions:
  - init_db(): creates DB table.
  - save_message(): inserts + prunes.
  - get_last_messages(): returns last N messages.
  - clear_user_memory(): deletes all messages of a user.

### 2. chatbot_retriever.py — Hybrid Retriever
- Loads PDFs and PPTX files.
- Automatically downloads files from HuggingFace dataset `07Codex07/PrepGraph-Data`.
- Splits files into overlapping chunks using RecursiveCharacterTextSplitter.
- Builds:
  - BM25 index (keyword search)
  - FAISS IVF index (semantic search)
  - SentenceTransformer embeddings (MiniLM-L6-v2)
- Retrieval metrics:
  - Chunk size: 400
  - Overlap: 80
  - Top K: 3
  - Max returned context: 4000 chars
  - FAISS nlist=100, nprobe=10, batch=256
- Key functions:
  - ensure_data_dir(): downloads dataset files.
  - load_all_docs(): loads and tags PDF pages.
  - build_or_load_indexes(): builds BM25 + FAISS + caches.
  - hybrid_retrieve(): merges BM25 + FAISS results.
  - retrieve_node_from_rows(): extracts latest user query and retrieves context.

### 3. chatbot_graph.py — LLM Logic (No UI)
- Contains the main system prompt.
- Prepares prompt messages: system prompt + last user messages + retrieved context.
- Calls Groq Llama-3.1 using ChatGroq.
- Cleans the output using _extract_answer_from_response().
- This file contains only logic and is used by FastAPI.

### 4. main_api.py — FastAPI Backend (Production)
- Exposes REST endpoints for frontend integration.
- Chat pipeline:
  1. Save user message to SQLite.
  2. Fetch last messages from DB.
  3. Extract latest user query.
  4. Retrieve context from the hybrid retriever.
  5. Trim context using tiktoken to avoid overflows.
  6. Build system prompt with RAG context.
  7. Call LLM (Groq).
  8. Extract clean answer.
  9. Save assistant message to SQLite.
  10. Return JSON to frontend.

- API Endpoints:
  - POST /chat → main chat endpoint.
  - GET /retrieve → test retriever directly.
  - POST /reindex → rebuild FAISS/BM25.
  - POST /upload → upload new PDF/PPTX.
  - GET /docs_list → list available documents.
  - GET /history/{user_id} → conversation memory.
  - POST /memory/clear → clear stored messages.

- CORS enabled for localhost:5173 (React dev server).
- servable static frontend if placed in /frontend/dist.

### 5. Dockerfile — Containerized Deployment
- Installs all requirements.
- Configures environment.
- Runs FastAPI with Uvicorn in production mode.
- Exposes port 8000.
- Designed for both local and cloud deployment.

---

## How /chat Works Internally
1. User sends message to /chat.
2. System saves the message to SQLite.
3. Fetches last messages (conversation history).
4. Extracts only the latest user query.
5. Runs hybrid_retrieve() → returns merged BM25 + FAISS chunks.
6. Trims context using a token limiter.
7. Builds final system prompt including context.
8. Calls Groq Llama-3.1 model.
9. Extracts plain answer text.
10. Saves assistant reply to SQLite.
11. Returns JSON with assistant reply + conversation history.

---

## Running the Server Without Docker
export GROQ_API_KEY="your_key"

pip install -r requirements.txt
uvicorn main_api:app --reload --port 8000

Server runs at: http://localhost:8000

---

## Docker Deployment
docker build -t prepgraph .
docker run -p 8000:8000 prepgraph

---

## Frontend Integration Example (React)
fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    user_id: "vinayak",
    message: "Explain deadlock"
  })
})
  .then(res => res.json())
  .then(data => console.log(data.assistant));

---

## Why PrepGraph Is Special
- Hybrid retrieval (BM25 + FAISS) for maximum accuracy.
- Automatic dataset download and indexing.
- Highly optimized prompt building.
- Persistent long-term memory.
- FastAPI JSON endpoints (frontend-ready).
- Fully dockerized.
- Production-grade caching and FAISS index persistence.
- Designed for academic chat + document Q/A.

