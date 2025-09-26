# semantic_search/schemas.py
from pydantic import BaseModel
from typing import List
from datetime import datetime

class UploadResponse(BaseModel):
    message: str
    doc_id: int
    chunks: int

class SearchResult(BaseModel):
    doc_id: int
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class DocumentResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    num_chunks: int
