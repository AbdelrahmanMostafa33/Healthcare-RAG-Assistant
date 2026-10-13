"""Reproducible evaluation-readiness check for notebooks/04_evaluation.ipynb.

Answers, in order:
1. Are the prerequisite data/index/eval files present?
2. Does config.validation accept the saved threshold?
3. What fingerprint field, if any, makes the saved threshold stale?
4. Does the RAG pipeline load if the index is currently compatible?
5. Which .env credentials are actually configured (values redacted)?

This script does not call any provider API. It is intended to be run from the
repository root with `python evaluate_readiness.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

def index_manifest_expectations() -> dict[str, object]:
    """Configuration values the index manifest is expected to record for this build.

    eval_sha256 and index_manifest_sha256 are deliberately absent: they belong to the threshold
    fingerprint, not to the index build, and a manifest cannot contain the hash of itself, so
    comparing them here would report a mismatch that could never clear.
    """
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
        "reranker_model": config.RERANKER_MODEL,
        "reranker_model_revision": config.RERANKER_MODEL_REVISION,
        "chunk_max_chars": config.CHUNK_MAX_CHARS,
        "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
        "model_runtime_versions": config.model_runtime_versions(),
    }


def present(paths: dict[str, Path]) -> dict[str, bool]:
    return {name: path.exists() for name, path in paths.items()}


def sha256_ok(path: Path) -> bool:
    return config.sha256_file(path) is not None


def _read_json(path: Path):
    """Return parsed JSON, or None when the file is absent or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def threshold_fingerprint_report() -> tuple[list[str], list[str]]:
    """Return (hard, soft) fields where the saved threshold no longer matches this build.

    This is the same comparison config.load_threshold() makes, so the hard fields listed here are
    exactly the ones that make a saved calibration stale. A constant such as safety_rules_version
    is covered because it is part of config.calibration_metadata().
    """
    saved = _read_json(config.THRESHOLD_PATH)
    if not isinstance(saved, dict):
        return [f"THRESHOLD_PATH missing or unreadable: {config.THRESHOLD_PATH}"], []

    expected = config.calibration_metadata()
    hard, soft = config.fingerprint_mismatches(saved, expected)
    # load_threshold() fails closed when a fingerprint value cannot be computed at all.
    uncomputable = sorted(key for key, value in expected.items() if value is None)
    if uncomputable:
        hard = [
            f"{key} could not be computed, so load_threshold() fails closed" for key in uncomputable
        ] + hard
    return hard, soft


def manifest_drift_report() -> list[str]:
    """Return index-manifest fields that disagree with the current code configuration."""
    manifest = _read_json(config.INDEX_MANIFEST_PATH)
    if not isinstance(manifest, dict):
        return [f"INDEX_MANIFEST_PATH missing or unreadable: {config.INDEX_MANIFEST_PATH}"]

    expected = index_manifest_expectations()
    return [
        f"{key}: config={expected[key]!r} manifest={manifest.get(key)!r}"
        for key in expected
        if expected[key] != manifest.get(key)
    ]


def credentials_report() -> tuple[dict[str, int], dict[str, str]]:
    """Return (counts, status text) per credential variable. Values are never read into the output."""
    from src.provider_clients import api_keys

    counts: dict[str, int] = {}
    statuses: dict[str, str] = {}
    for env in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
        try:
            keys = api_keys(env)
            counts[env] = len(keys)
            statuses[env] = f"{len(keys)} key(s) configured"
        except RuntimeError as error:
            counts[env] = 0
            statuses[env] = f"not configured: {error}"
    return counts, statuses


def pipeline_load_report() -> dict[str, str | bool]:
    from src.pipeline import load_pipeline

    try:
        pipeline = load_pipeline()
        return {
            "ok": True,
            "threshold": pipeline.threshold,
            "calibrated": pipeline.calibrated,
            "chunks": pipeline.retriever.index.ntotal,
        }
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    print("=== evaluation readiness ===\n")

    paths = {
        "EVAL_PATH": config.EVAL_PATH,
        "INDEX_PATH": config.INDEX_PATH,
        "CHUNKS_PATH": config.CHUNKS_PATH,
        "INDEX_MANIFEST_PATH": config.INDEX_MANIFEST_PATH,
        "THRESHOLD_PATH": config.THRESHOLD_PATH,
        "KB_PATH": config.KB_PATH,
    }
    files_present = present(paths)
    checksums_ok = {name: sha256_ok(path) for name, path in paths.items()}

    print("1. Prerequisite files")
    for name, ok in files_present.items():
        print(f"   {'OK ' if ok else 'MISSING'} {name} = {paths[name]}")
    print(f"   ALL_FILES_EXIST = {all(files_present.values())}\n")

    print("2. File checksums reachable")
    for name, ok in checksums_ok.items():
        print(f"   {'OK ' if ok else 'FAIL'} {name} sha256 reachable = {ok}")
    print(f"   SAFETY_RULES_VERSION = {config.SAFETY_RULES_VERSION}\n")

    hard, soft = threshold_fingerprint_report()
    print("3. Threshold validation / fingerprint mismatches")
    print(f"   load_threshold() = {config.load_threshold()}")
    print(f"   DEFAULT_THRESHOLD = {config.DEFAULT_THRESHOLD}")
    if hard:
        print("   STALE_FINGERPRINT_FIELDS =")
        for line in hard:
            print(f"     - {line}")
    else:
        print("   STALE_FINGERPRINT_FIELDS = []")
    if soft:
        print("   SOFT_FINGERPRINT_WARNINGS =")
        for line in soft:
            print(f"     - {line}")

    drift = manifest_drift_report()
    if drift:
        print("   MANIFEST_DRIFT =")
        for line in drift:
            print(f"     - {line}")
    else:
        print("   MANIFEST_DRIFT = []")
    print()

    print("4. Pipeline load report")
    load_report = pipeline_load_report()
    for key, value in load_report.items():
        print(f"   {key} = {value}")
    print()

    counts, statuses = credentials_report()
    print("5. Credentials configured (values redacted)")
    for env, status in statuses.items():
        print(f"   {env}: {status}")
    print()

    ready = (
        all(files_present.values())
        and all(checksums_ok.values())
        and config.load_threshold()[1]
        and bool(load_report["ok"])
        and all(count > 0 for count in counts.values())
    )
    print(f"READY_FOR_FULL_EVAL = {ready}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
