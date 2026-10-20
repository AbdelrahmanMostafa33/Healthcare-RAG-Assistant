"""Project settings and validation of the calibrated abstention threshold."""
import hashlib
import json
import logging
import math
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

from src.safety import SAFETY_RULES_VERSION

logger = logging.getLogger(__name__)

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

MEDQUAD_DATASET_ID = "lavita/MedQuAD"
MEDQUAD_REVISION = "84ea67f83cec9692ad254eaa02c9731b24ecfe4c"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
GENERATOR_PROVIDER = os.getenv("GENERATOR_PROVIDER", "gemini").strip().lower()
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq").strip().lower()
_DEFAULT_JUDGE_MODEL = "gpt-oss-120b" if JUDGE_PROVIDER == "cerebras" else "openai/gpt-oss-120b"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _DEFAULT_JUDGE_MODEL).strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").strip()

RETRIEVE_K = 40
CONTEXT_K = 5
RAG_PIPELINE_VERSION = "source-diverse-context-v6"
MAX_TOKENS = 2048

# The timeout is a ceiling for a busy provider, not a target. A slower generator can still be
# selected with GENERATOR_MODEL: gemini-3.8-flash was measured at 58-131 s per call, and an earlier
# 60 s client timeout turned that ordinary queuing into a 504 that looked like a provider failure.
# The default gemini-3.5-flash-lite answered a five-passage prompt in about a second, so the
# generous ceiling costs it nothing.
GENERATOR_TIMEOUT_MS = int(os.getenv("GENERATOR_TIMEOUT_MS", "240000"))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Generation is pinned to a deterministic temperature for evaluation, where an answer must be
# reproducible against a frozen threshold. Override it only for interactive demos.
GENERATOR_TEMPERATURE = _env_float("GENERATOR_TEMPERATURE", 0.0)

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


def sha256_text_file(path: Path) -> str | None:
    """SHA-256 of a text file with CRLF and CR normalised to LF, or None when unavailable.

    A fingerprint hashes files that git may check out with either line ending, so the stored hash
    must not depend on the platform that produced it.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def model_runtime_versions() -> dict[str, str | None]:
    """Record library versions that can change encoder and reranker scores."""
    versions = {}
    for package in ("sentence-transformers", "transformers", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


# Runtime-library versions can drift with a patch release without changing a pinned model. They are
# reported as a warning instead of rejecting a saved threshold; everything else is a hard match.
SOFT_FINGERPRINT_FIELDS = frozenset({"model_runtime_versions"})


def calibration_metadata() -> dict:
    """Fingerprints of everything that makes a saved reranker threshold meaningful."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "reranker_model": RERANKER_MODEL,
        "reranker_model_revision": RERANKER_MODEL_REVISION,
        "rag_pipeline_version": RAG_PIPELINE_VERSION,
        "model_runtime_versions": model_runtime_versions(),
        "eval_sha256": sha256_text_file(EVAL_PATH),
        "index_manifest_sha256": sha256_text_file(INDEX_MANIFEST_PATH),
        "safety_rules_version": SAFETY_RULES_VERSION,
    }


def fingerprint_mismatches(saved: dict, expected: dict) -> tuple[list[str], list[str]]:
    """Split fingerprint differences into hard mismatches and warnings."""
    hard, soft = [], []
    for key, value in expected.items():
        if saved.get(key) == value:
            continue
        (soft if key in SOFT_FINGERPRINT_FIELDS else hard).append(key)
    return hard, soft


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
        hard, soft = fingerprint_mismatches(saved, expected)
        if soft:
            logger.warning(
                "Stored calibration was computed with different runtime versions (%s); "
                "recalibrate if retrieval scores have shifted.",
                ", ".join(sorted(soft)),
            )
        if hard:
            return DEFAULT_THRESHOLD, False
        return threshold, True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return DEFAULT_THRESHOLD, False
