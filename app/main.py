# app/main.py
from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Semantic Search API",
    description="Upload PDFs, build FAISS index, and search with semantic similarity",
    version="1.0.0",
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API. Visit /docs for Swagger UI."}