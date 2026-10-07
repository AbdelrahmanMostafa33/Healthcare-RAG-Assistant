"""Settings in one place. The only environment variable is GROQ_API_KEY (see .env.example)."""
import json
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
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
LLM_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "qwen/qwen3.8-27b"   # grades answers in notebook 04, never the generator
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
