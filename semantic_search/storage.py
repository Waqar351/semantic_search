# semantic_search/storage.py
import os
from fastapi import UploadFile

def save_upload(file: UploadFile, filename: str) -> str:
    """Save uploaded file to /uploads and return its path"""
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path
