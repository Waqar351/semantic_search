# semantic_search/core.py
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
import numpy as np

class SemanticSearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def load_pdf(self, path, chunk_size=10):
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        words = text.split()
        self.chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        # breakpoint()
        return self.chunks

    def build_index(self):
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        # breakpoint()
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        dists, idxs = self.index.search(q_emb, top_k)
        results = [(self.chunks[int(i)], float(d)) for i, d in zip(idxs[0], dists[0])]
        breakpoint()
        return results
