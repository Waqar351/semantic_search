# semantic_search/cli_advanced.py
import nltk

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

import argparse
from app.core_2 import SemanticSearchAdvanced
import os
import glob

parser = argparse.ArgumentParser(description="Advanced Semantic Search CLI (Multiple PDFs)")
parser.add_argument("--pdf", nargs="+", help="One or more PDF files")
parser.add_argument("--pdf_folder", help="Folder containing PDF files")
parser.add_argument("--k", type=int, default=5, help="Number of top results to return")
parser.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranker for better relevance")
args = parser.parse_args()

# Collect PDF files
pdf_files = []
if args.pdf:
    pdf_files.extend(args.pdf)
if args.pdf_folder:
    pdf_files.extend(glob.glob(os.path.join(args.pdf_folder, "*.pdf")))

if not pdf_files:
    print("No PDFs found. Use --pdf or --pdf_folder.")

# Initialize semantic search
ss = SemanticSearchAdvanced(
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2" if args.rerank else None
)

# Load all PDFs
total_chunks = 0
for pdf in pdf_files:
    chunks = ss.load_pdf(pdf)
    total_chunks += len(chunks)
print(f"Loaded {total_chunks} chunks from {len(pdf_files)} PDFs.")

# Build FAISS index
ss.build_index()
print("Index built successfully.\n")

# Interactive search loop
while True:
    query = input("Enter your query (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("Exiting...")
        break

    results = ss.search(query, top_k=args.k, rerank=args.rerank)

    for idx, r in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"Hybrid Score: {r['hybrid_score']:.4f}")
        print(f"Semantic Score: {r['semantic_score']:.4f}")
        print(f"Keyword Score: {r['keyword_score']:.4f}")
        if args.rerank and "rerank_score" in r:
            print(f"Rerank Score: {r['rerank_score']:.4f}")
        print(f"PDF: {r['metadata']['pdf']} | Page: {r['metadata']['page']}")
        print(f"Preview:\n{r['preview']}\n")
    print("="*100)

