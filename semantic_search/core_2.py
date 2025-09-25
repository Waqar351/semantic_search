# semantic_search/core_advanced_v2.py
import os
import pickle
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# -----------------------------
# Robust sentence tokenizer
# -----------------------------
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except (LookupError, ModuleNotFoundError):
    import nltk
    nltk.download('punkt', quiet=True)
from nltk.tokenize import PunktSentenceTokenizer

class SemanticSearchAdvanced:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", reranker_model=None):
        """
        Advanced Semantic Search with cosine similarity + hybrid scoring.
        """
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        self.tokenizer = PunktSentenceTokenizer()

    # -----------------------------
    # PDF Loading & Chunking
    # -----------------------------
    def load_pdf(self, path, chunk_size=5, overlap=2, min_chars=20):
        """
        Load PDF and split into overlapping sentence chunks.
        """
        chunks = []
        metadata = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                sentences = self.tokenizer.tokenize(text)
                for i in range(0, len(sentences), chunk_size - overlap):
                    chunk = " ".join(sentences[i:i+chunk_size]).strip()
                    if len(chunk) >= min_chars:
                        chunks.append(chunk)
                        metadata.append({
                            "pdf": os.path.basename(path),
                            "page": page_num + 1,
                            "chunk_idx": len(chunks)-1
                        })

        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        return chunks

    # -----------------------------
    # Build FAISS Index (cosine similarity)
    # -----------------------------
    def build_index(self, use_ivf=False, nlist=100):
        """
        Build FAISS index with normalized embeddings for cosine similarity.
        """
        if not self.chunks:
            raise ValueError("No chunks loaded.")

        embeddings = self.model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        # Normalize for cosine similarity
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

    # -----------------------------
    # Save / Load Index
    # -----------------------------
    def save_index(self, path="faiss_index.faiss", metadata_path="metadata.pkl"):
        faiss.write_index(self.index, path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self, path="faiss_index.faiss", metadata_path="metadata.pkl"):
        self.index = faiss.read_index(path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    # -----------------------------
    # Preview text snippet
    # -----------------------------
    def _get_preview(self, chunk, query, window=50):
        idx = chunk.lower().find(query.lower())
        if idx == -1:
            return chunk[:200] + "..."
        start = max(0, idx - window)
        end = min(len(chunk), idx + window)
        return chunk[start:end] + "..."

    # -----------------------------
    # Keyword match score
    # -----------------------------
    def _keyword_score(self, query, chunk):
        query_terms = query.lower().split()
        chunk_lower = chunk.lower()
        matches = sum(1 for term in query_terms if term in chunk_lower)
        return matches / max(len(query_terms), 1)

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, query, top_k=5, rerank=False, alpha=0.7):
        """
        Search top_k chunks.
        alpha: weight for semantic similarity in hybrid scoring
        """
        if self.index is None:
            raise ValueError("Index not built.")

        # Encode query & normalize
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # Search
        dists, idxs = self.index.search(q_emb, top_k*3)  # retrieve extra for reranking

        results = []
        for i, d in zip(idxs[0], dists[0]):
            chunk_text = self.chunks[int(i)]
            sem_score = float(d)
            kw_score = self._keyword_score(query, chunk_text)
            hybrid_score = alpha * sem_score + (1-alpha) * kw_score
            results.append({
                "chunk": chunk_text,
                "semantic_score": sem_score,
                "keyword_score": kw_score,
                "hybrid_score": hybrid_score,
                "metadata": self.metadata[int(i)]
            })

        # Optional reranking
        if rerank and self.reranker:
            pairs = [(query, r["chunk"]) for r in results]
            rerank_scores = self.reranker.predict(pairs)
            for r, s in zip(results, rerank_scores):
                r["rerank_score"] = float(s)
            results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        else:
            results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

        # Add preview
        for r in results:
            r["preview"] = self._get_preview(r["chunk"], query)

        return results[:top_k]
