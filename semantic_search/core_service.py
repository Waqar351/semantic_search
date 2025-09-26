# app/core_service.py
import os
import pickle
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from nltk.tokenize import PunktSentenceTokenizer
import nltk
from typing import List, Dict

# # Ensure punkt available
# try:
#     nltk.data.find('tokenizers/punkt')
# except (LookupError, ModuleNotFoundError):
#     nltk.download('punkt', quiet=True)

class SemanticSearchService:
    """
    Core service: load PDFs, chunk, embed, build FAISS index (cosine),
    hybrid scoring (semantic + keyword) and optional reranking.
    Persists index and metadata to disk.
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 reranker_model: str | None = None,
                 storage_dir: str = "data"):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None
        self.tokenizer = PunktSentenceTokenizer()
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []   # dicts with keys: pdf, page, chunk_idx, doc_id
        self.embeddings = None
        self.index = None
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "faiss.index")
        self.meta_path = os.path.join(self.storage_dir, "metadata.pkl")

    def reset(self):
        """Clear current chunks, metadata, embeddings, and index."""
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        self.index = None

    def load_pdf(self, path: str, doc_id: int,
             chunk_size: int = 5, overlap: int = 1, min_chars: int = 20, reset: bool = True) -> int:
        """
        Read a PDF and add its chunks to internal lists.
        Returns the number of chunks added.
        doc_id ties chunks to a document in external DB.
        Handles short pages and prevents duplicate chunks.
        """
        if reset:
            self.reset()
        added = 0
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                sentences = self.tokenizer.tokenize(text)
                if not sentences:
                    continue

                step = max(1, chunk_size - overlap)
                i = 0
                while i < len(sentences):
                    chunk_sentences = sentences[i:i+chunk_size]
                    if not chunk_sentences:
                        break
                    chunk = " ".join(chunk_sentences).strip()
                    if len(chunk) >= min_chars and chunk not in self.chunks:
                        chunk_idx = len(self.chunks)
                        self.chunks.append(chunk)
                        self.metadata.append({
                            "pdf": os.path.basename(path),
                            "page": page_num + 1,
                            "chunk_idx": chunk_idx,
                            "doc_id": doc_id
                        })
                        added += 1
                    i += step
        return added


    # Build or rebuild FAISS index
    def build_index(self, use_ivf: bool = False, nlist: int = 100):
        if not self.chunks:
            raise ValueError("No chunks to index.")
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        # normalize for cosine (inner product on unit vectors)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings
        d = embeddings.shape[1]

        if use_ivf and len(self.chunks) > 1:
            nlist = min(nlist, len(self.chunks))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(embeddings)
        else:
            index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        self.index = index

    # Persistence
    def save(self):
        if self.index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                obj = pickle.load(f)
            self.chunks = obj["chunks"]
            self.metadata = obj["metadata"]
            # load embeddings if you stored them (optional)
            # for simplicity we won't reload embeddings into memory here

    # Utility: preview and keyword score
    def _get_preview(self, chunk: str, query: str, window: int = 80) -> str:
        q = query.lower()
        idx = chunk.lower().find(q)
        if idx == -1:
            return (chunk[:200] + "...") if len(chunk) > 200 else chunk
        start = max(0, idx - window)
        end = min(len(chunk), idx + window)
        return ("..." if start > 0 else "") + chunk[start:end] + ("..." if end < len(chunk) else "")

    def _keyword_score(self, query: str, chunk: str) -> float:
        q_terms = [t for t in query.lower().split() if t.strip()]
        if not q_terms:
            return 0.0
        chunk_low = chunk.lower()
        matches = sum(1 for t in q_terms if t in chunk_low)
        return matches / len(q_terms)

    # Search
    def search(self, query: str, top_k: int = 10, rerank: bool = False, alpha: float = 0.75):
        if self.index is None:
            raise ValueError("Index not built.")
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        # fetch more to allow hybrid/rerank
        fetch_k = min(len(self.chunks), max(top_k * 3, top_k + 10))
        dists, idxs = self.index.search(q_emb, fetch_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            chunk = self.chunks[int(idx)]
            sem_score = float(dist)  # cosine-like: higher = more similar
            kw = self._keyword_score(query, chunk)
            hybrid = alpha * sem_score + (1 - alpha) * kw
            results.append({
                "chunk": chunk,
                "semantic_score": sem_score,
                "keyword_score": kw,
                "hybrid_score": hybrid,
                "metadata": self.metadata[int(idx)]
            })
        # optional rerank with CrossEncoder
        if rerank and self.reranker:
            pairs = [(query, r["chunk"]) for r in results]
            rerank_scores = self.reranker.predict(pairs)
            for r, s in zip(results, rerank_scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda r: r["rerank_score"], reverse=True)
        else:
            results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        # attach previews
        for r in results:
            r["preview"] = self._get_preview(r["chunk"], query)
        return results[:top_k]
    
# Global service instance
semantic_search_service = SemanticSearchService()
semantic_search_service.load()  # Load existing index + metadata if available