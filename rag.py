import fitz
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SimpleRAG:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.index = None
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_splitter()

    def _build_splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def update_chunk_params(self, chunk_size, chunk_overlap):
        """Rebuild splitter and re-chunk/re-embed if params changed."""
        if chunk_size != self.chunk_size or chunk_overlap != self.chunk_overlap:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self._build_splitter()
            return True
        return False

    def load_pdf(self, path: str) -> str:
        doc = fitz.open(path)
        return "\n\n".join(page.get_text() for page in doc)

    def load_pdf_bytes(self, file_bytes) -> str:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc)

    def ingest(self, text: str):
        self.chunks = self.splitter.split_text(text)
        embeddings = self.embedder.encode(self.chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype("float32"))
        return self.chunks

    def retrieve(self, query: str, k: int = 3):
        query_embedding = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        return retrieved_chunks, distances[0], indices[0]

    def build_prompt(self, question: str, context_chunks: list) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        return (
            f"Answer based only on the context provided. "
            f"If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

    def ask(self, question: str, k: int = 3, model: str = "tinyllama") -> dict:
        chunks, distances, indices = self.retrieve(question, k)
        prompt = self.build_prompt(question, chunks)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            answer = response.json().get("response", "Error: no response from model")
        except requests.exceptions.ConnectionError:
            answer = "Error: Could not connect to Ollama. Make sure it's running."
        except requests.exceptions.Timeout:
            answer = "Error: Request timed out. The model may be too slow on this machine."

        return {
            "answer": answer,
            "chunks": chunks,
            "distances": distances.tolist(),
            "indices": indices.tolist(),
            "prompt": prompt,
            "total_chunks": len(self.chunks),
            "k": k
        }