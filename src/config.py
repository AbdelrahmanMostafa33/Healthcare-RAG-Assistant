"""Project settings and validation of the calibrated abstention threshold."""
import hashlib
import json
import math
import os
from importlib.metadata import PackageNotFoundError, version
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
SAFETY_POLICY_PATH = Path(__file__).with_name("safety.py")

CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

MEDQUAD_DATASET_ID = "lavita/MedQuAD"
MEDQUAD_REVISION = "84ea67f83cec9692ad254eaa02c9731b24ecfe4c"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
GENERATOR_PROVIDER = os.getenv("GENERATOR_PROVIDER", "gemini").strip().lower()
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-3.8-flash").strip()
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq").strip().lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()

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


def model_runtime_versions() -> dict[str, str | None]:
    """Record library versions that can change encoder and reranker scores."""
    versions = {}
    for package in ("sentence-transformers", "transformers", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def calibration_metadata() -> dict:
    """Fingerprints of everything that makes a saved reranker threshold meaningful."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "reranker_model": RERANKER_MODEL,
        "reranker_model_revision": RERANKER_MODEL_REVISION,
        "model_runtime_versions": model_runtime_versions(),
        "eval_sha256": sha256_file(EVAL_PATH),
        "index_manifest_sha256": sha256_file(INDEX_MANIFEST_PATH),
        "safety_rules_sha256": sha256_file(SAFETY_POLICY_PATH),
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
