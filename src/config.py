"""Settings in one place.

Environment variables (see .env):
  GROQ_API_KEY   - one or more Groq keys, comma-separated (generator + Groq-judge)
  GEMINI_API_KEY - one Google AI Studio key (Gemini judge only)
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

# Judge: a different model family from the generator.
# - Groq judge: qwen/qwen3.8-27b (OpenAI-compatible, reached through the Groq client)
# - Gemini judge: gemini-3.5-flash-lite (Google API, separate GEMINI_API_KEY)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
JUDGE_IS_GEMINI = JUDGE_MODEL.startswith("gemini-")
LLM_BASE_URL = "https://api.groq.com/openai/v1"

RETRIEVE_K = 20   # candidates from FAISS
CONTEXT_K = 5     # passages kept after reranking and put in the prompt
MAX_TOKENS = 2048

# The reranker returns raw logits; 0 means "more likely relevant than not".
# Placeholder until notebook 04 calibrates a real value on the dev questions.
DEFAULT_THRESHOLD = 0.0

# Emergency detection runs before retrieval. Turn off only for testing.
EMERGENCY_ENABLED = True


def load_threshold():
    """Returns (threshold, calibrated)."""
    if THRESHOLD_PATH.exists():
        return json.loads(THRESHOLD_PATH.read_text())["threshold"], True
    return DEFAULT_THRESHOLD, False
