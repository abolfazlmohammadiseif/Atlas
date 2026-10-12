"""
FastAPI application for the RAG pipeline.
Exposes endpoints for document ingestion and question answering.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline
from src.utils.logger import logger

pipeline = RAGPipeline()

app = FastAPI(
    title="RAG Application API",
    description="API for ingesting documents and asking questions.",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    """Schema for document ingestion request."""

    text: str
    metadata: dict[str, str] | None = None


class AskRequest(BaseModel):
    """Schema for question asking request."""

    question: str


@app.post("/ingest")
def ingest_document(request: IngestRequest) -> dict[str, str]:
    """Ingests raw text into the vector database."""
    try:
        pipeline.ingest_text(text=request.text, metadata=request.metadata)
        return {"status": "success", "message": "Document ingested successfully."}
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Internal server error during ingestion."
        ) from e


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, str]:
    """Answers a question based on ingested documents."""
    try:
        answer = pipeline.ask(question=request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error("Question answering failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Internal server error during generation."
        ) from e
