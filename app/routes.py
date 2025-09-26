# app/routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from semantic_search.core_service import semantic_search_service
import os
import shutil

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), doc_id: int = Form(0)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        added = semantic_search_service.load_pdf(file_path, doc_id=doc_id)
        semantic_search_service.build_index()
        semantic_search_service.save()

        return {"status": "success", "chunks_added": added}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/")
async def search(query: str, top_k: int = 5, rerank: bool = False):
    try:
        results = semantic_search_service.search(query, top_k=top_k, rerank=rerank)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
