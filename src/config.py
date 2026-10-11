"""Project settings and validation of the calibrated abstention threshold."""
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KB_PATH = ROOT / "data" / "processed" / "kb_documents.csv"
INDEX_PATH = ROOT / "data" / "index" / "kb.faiss"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"
INDEX_MANIFEST_PATH = ROOT / "data" / "index" / "manifest.json"
EVAL_PATH = ROOT / "data" / "eval" / "questions.csv"
REPORTS_DIR = ROOT / "reports"
THRESHOLD_PATH = REPORTS_DIR / "abstention_threshold.json"

CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
# This Groq model is currently a preview model; choose another supported JSON-capable judge via .env if needed.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")

RETRIEVE_K = 20
CONTEXT_K = 5
MAX_TOKENS = 2048

# A numeric fallback is retained for backwards-compatible result reporting, but is NEVER trusted
# to answer questions. RAGPipeline abstains until the threshold has been calibrated for this build.
DEFAULT_THRESHOLD = 0.0


def sha256_file(path: Path) -> str | None:
    """Return a file's SHA-256, or None when it is not available."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def calibration_metadata() -> dict:
    """Fingerprints of everything that makes a saved reranker threshold meaningful."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "eval_sha256": sha256_file(EVAL_PATH),
        "index_manifest_sha256": sha256_file(INDEX_MANIFEST_PATH),
    }


def load_threshold():
    """Return (threshold, calibrated); stale, malformed or incomplete files fail closed."""
    if not THRESHOLD_PATH.exists():
        return DEFAULT_THRESHOLD, False
    try:
        saved = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        raw_threshold = saved["threshold"]
        if isinstance(raw_threshold, bool):
            return DEFAULT_THRESHOLD, False
        threshold = float(raw_threshold)
        if not math.isfinite(threshold):
            return DEFAULT_THRESHOLD, False
        expected = calibration_metadata()
        if any(value is None for value in expected.values()):
            return DEFAULT_THRESHOLD, False
        if any(saved.get(key) != value for key, value in expected.items()):
            return DEFAULT_THRESHOLD, False
        return threshold, True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return DEFAULT_THRESHOLD, False
