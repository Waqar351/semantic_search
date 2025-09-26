# semantic_search/cli.py
import argparse
from app.core import SemanticSearch
from PyPDF2 import PdfReader


# def main():
parser = argparse.ArgumentParser()
parser.add_argument("--pdf", required=True, help="Path to PDF")
parser.add_argument("--k", type=int, default=3)
args = parser.parse_args()

# reader = PdfReader("data/DocumentoTokioMarine_pdf.pdf")
# breakpoint
# for page in reader.pages:
#     print(page.extract_text())

# breakpoint()

ss = SemanticSearch()
chunks = ss.load_pdf(args.pdf)
print(f"Loaded {len(chunks)} chunks from PDF.")

# breakpoint()
ss.build_index()

while True:
    q = input("Ask (or 'exit'): ")
    if q.strip().lower() == "exit":
        break
    for text, score in ss.search(q, top_k=args.k):
        print(f"\nScore: {score:.4f}\n{text[:400]}...\n")
