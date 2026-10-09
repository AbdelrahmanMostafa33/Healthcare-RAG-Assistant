"""Settings in one place.

Environment variables (see .env.example):
  GROQ_API_KEY - Groq API key, used for the generator and the evaluation judge
                 (one key; if comma-separated, only the first key is used)
  JUDGE_MODEL  - optional, overrides the judge model below
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KB_PATH = ROOT / "data" / "processed" / "kb_documents.csv"
INDEX_PATH = ROOT / "data" / "index" / "kb.faiss"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"
EVAL_PATH = ROOT / "data" / "eval" / "questions.csv"
REPORTS_DIR = ROOT / "reports"
THRESHOLD_PATH = REPORTS_DIR / "abstention_threshold.json"

CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_BASE_URL = "https://api.groq.com/openai/v1"

# The judge must be a different model family from the generator, so it does not grade its own style.
# It runs on Groq through the same OpenAI-compatible client.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")

RETRIEVE_K = 20   # candidates from FAISS
CONTEXT_K = 5     # passages kept after reranking and put in the prompt
MAX_TOKENS = 2048

# The reranker returns raw logits; 0 means "more likely relevant than not".
# Placeholder until notebook 04 calibrates a real value on the dev questions.
DEFAULT_THRESHOLD = 0.0


def load_threshold():
    """Returns (threshold, calibrated).

    A saved threshold only means something for the reranker it was calibrated with, so one saved
    for a different reranker is ignored and the placeholder is used instead.
    """
    if THRESHOLD_PATH.exists():
        saved = json.loads(THRESHOLD_PATH.read_text())
        if saved.get("reranker_model", RERANKER_MODEL) == RERANKER_MODEL:
            return saved["threshold"], True
    return DEFAULT_THRESHOLD, False
