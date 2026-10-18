"""FastAPI service. Run with: uvicorn app:app

GET  /          ->  demo page
POST /query     {"question": "..."}  ->  answer, safety flags, sources, disclaimer
GET  /health    ->  startup and calibration state
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("healthcare_rag")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def _load_pipeline():
    # Import torch before src.kb loads pandas; this avoids a native DLL conflict on Windows.
    import torch

    logger.info("Loading retrieval models on %s", "CUDA" if torch.cuda.is_available() else "CPU")
    from src.pipeline import load_pipeline

    return load_pipeline()


async def _initialize_pipeline(app):
    try:
        app.state.pipeline = await asyncio.to_thread(_load_pipeline)
    except Exception:
        app.state.startup_error = True
        logger.exception("Could not initialize the healthcare assistant")


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=1000, examples=["What are the symptoms of type 2 diabetes?"])


class Source(BaseModel):
    n: int   # matches the [n] citations in the answer
    source: str
    title: str
    url: str
    excerpt: str = ""


class QueryResponse(BaseModel):
    answer: str
    abstained: bool   # True when no generated answer was returned to the user
    # Why it was withheld: "insufficient_evidence" (the threshold), "citation_validation_failed"
    # (the answer cited nothing usable) or "uncalibrated" (no trusted threshold yet).
    abstain_reason: str | None = None
    emergency: bool   # True when the question described an emergency (fixed message, no retrieval, no LLM)
    redirected: bool  # True when the question asked for personal medical advice (fixed redirect to a doctor)
    refused: bool     # True when the question requested harmful or deceptive instructions
    sources: list[Source]
    disclaimer: str


@asynccontextmanager
async def lifespan(app):
    # Keep the demo page available while local model weights load in the background.
    app.state.pipeline = None
    app.state.startup_error = False
    initialization = asyncio.create_task(_initialize_pipeline(app))
    yield
    await initialization


app = FastAPI(
    title="Healthcare RAG Assistant",
    description="Educational health information from curated sources. Not medical advice.",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        return {
            "status": "error" if request.app.state.startup_error else "loading",
            "abstain_threshold": None,
            "threshold_calibrated": False,
        }
    return {
        "status": "ok" if pipeline.calibrated else "degraded",
        "abstain_threshold": pipeline.threshold,
        "threshold_calibrated": pipeline.calibrated,
    }


# A plain `def` endpoint runs in a thread pool, so a slow LLM call does not block other requests.
@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        if request.app.state.startup_error:
            raise HTTPException(status_code=503, detail="The assistant could not be initialized. Check the server logs.")
        raise HTTPException(status_code=503, detail="The assistant is still loading. Please try again shortly.")
    try:
        result = pipeline.answer(body.question)
        for source in result["sources"]:
            passage = result["passages"][source["n"] - 1]
            text = passage.get("text", "")
            source["excerpt"] = text.partition("\n")[2].strip() or text.strip()
        return result
    except Exception:
        logger.exception("query failed")
        raise HTTPException(status_code=502, detail="The answer could not be generated. Please try again.")
