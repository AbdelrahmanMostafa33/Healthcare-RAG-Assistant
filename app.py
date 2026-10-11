"""FastAPI service. Run with: uvicorn app:app

POST /query   {"question": "..."}  ->  answer, safety flags, sources, disclaimer
GET  /health
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("healthcare_rag")


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=1000, examples=["What are the symptoms of type 2 diabetes?"])


class Source(BaseModel):
    n: int   # matches the [n] citations in the answer
    source: str
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool   # True when the sources did not contain enough relevant evidence
    emergency: bool   # True when the question described an emergency (fixed message, no retrieval, no LLM)
    redirected: bool  # True when the question asked for personal medical advice (fixed redirect to a doctor)
    refused: bool     # True when the question requested harmful or deceptive instructions
    sources: list[Source]
    disclaimer: str


@asynccontextmanager
async def lifespan(app):
    # Import torch before src.kb loads pandas; this avoids a native DLL conflict on Windows.
    import torch

    logger.info("Loading retrieval models on %s", "CUDA" if torch.cuda.is_available() else "CPU")
    from src.pipeline import load_pipeline

    app.state.pipeline = load_pipeline()
    yield


app = FastAPI(
    title="Healthcare RAG Assistant",
    description="Educational health information from curated sources. Not medical advice.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    pipeline = request.app.state.pipeline
    return {
        "status": "ok" if pipeline.calibrated else "degraded",
        "abstain_threshold": pipeline.threshold,
        "threshold_calibrated": pipeline.calibrated,
    }


# A plain `def` endpoint runs in a thread pool, so a slow LLM call does not block other requests.
@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    try:
        return request.app.state.pipeline.answer(body.question)
    except Exception:
        logger.exception("query failed")
        raise HTTPException(status_code=502, detail="The answer could not be generated. Please try again.")
