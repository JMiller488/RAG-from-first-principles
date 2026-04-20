# RAG From Scratch

An interactive explorer for understanding Retrieval-Augmented Generation (RAG) from first principles.

## What This Is

A Streamlit app that lets you upload a PDF and ask it questions, while showing you **exactly** what happens at each step of the RAG pipeline. The purpose is to de-mistify RAG and to play around with paramaters to truly understand what is going on in under the hood:

1. **Chunking** — the document is split into overlapping text segments
2. **Embedding** — each chunk is mapped to a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Retrieval** — your question is embedded into the same vector space, and FAISS finds the `k` nearest chunks by L2 distance
4. **Prompt construction** — the retrieved chunks are concatenated as context alongside your question
5. **Generation** — a local LLM (via Ollama) produces an answer grounded in the retrieved context

You can interactively adjust chunk size, overlap, and `k` to see how each parameter affects retrieval quality and answer accuracy.

## How RAG Works

LLMs only know what was in their training data. RAG solves this by retrieving relevant passages from your own documents and injecting them into the prompt at inference time.

The key insight: **the LLM never sees vectors**. Vector search is a filtering step that selects which text chunks to include in the prompt. The model receives plain text — it has no idea that similarity search happened further upstream.

```
Documents → Chunks → Embeddings → FAISS Index
                                        ↑
Query → Embed → Nearest Neighbours → Text Chunks → Prompt → LLM → Answer
```

## Setup

```bash
# Clone and install
git clone <your-repo-url>
cd rag-from-scratch
python -m venv .venv
.venv\Scripts\Activate  # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Install and start Ollama (https://ollama.com)
ollama pull tinyllama  # or mistral if you have 8GB+ RAM

# Run
streamlit run app.py
```

## LLM Backend

The app defaults to **TinyLlama** via Ollama for cost and memory purposes. The LLM used here is not important. The same retrieved chunks fed to Claude or GPT-4 would produce significantly better answers - but the purpose of this project is not to get accurate answers - it's to easily understand the mechanics of RAG from first principles - so any LLM is fine.

To use a different Ollama model, just type its name in the sidebar (e.g. `mistral`, `llama3`, `phi3`).

## Tech Stack

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Vector search**: FAISS (IndexFlatL2 — exact nearest neighbours)
- **Text splitting**: `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- **PDF extraction**: PyMuPDF
- **LLM inference**: Ollama (local)
- **UI**: Streamlit

## Trade-offs & Limitations

- **TinyLlama (1.1B params)** often produces poor answers even with correct retrieval — this is intentional and documented as a teaching point
- **Chunk size** matters: tables and structured data can get split across chunks, losing context
- **Single-pass retrieval** may miss information spread across distant sections of a document
- No re-ranking, query decomposition, or hybrid search — this is a from-scratch implementation focused on clarity over production robustness