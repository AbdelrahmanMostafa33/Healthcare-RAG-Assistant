"""Calibrate the RAG abstention threshold without using the LLM API.

Run after building the processed knowledge base and FAISS index:
    python -m scripts.calibrate_threshold
"""
import json

import pandas as pd

from src.calibration import calibrate_threshold, save_calibration
from src import config
from src.retriever import load_retriever


def main():
    if not config.EVAL_PATH.exists():
        raise SystemExit(f"Evaluation questions not found: {config.EVAL_PATH}")
    questions = pd.read_csv(config.EVAL_PATH)
    print("Loading FAISS index and retrieval models (no GROQ_API_KEY is needed)...")
    retriever = load_retriever()
    result = calibrate_threshold(retriever, questions)
    save_calibration(result)
    print(json.dumps({
        "threshold": result["threshold"],
        "abstention_recall_dev": result["unanswerable_abstained"],
        "answerable_answered_dev": result["answerable_answered"],
        "dev_questions_scored": result["dev_questions_scored"],
        "saved_to": str(config.THRESHOLD_PATH.relative_to(config.ROOT)),
    }, indent=2))
    print("This is a dev-set calibration, not an independent performance claim. Freeze the value before test evaluation.")


if __name__ == "__main__":
    main()
