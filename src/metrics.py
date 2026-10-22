"""Small, deterministic metrics used by the evaluation notebook."""

import math

import numpy as np

from src.generator import citation_numbers
from src.pipeline import ABSTAIN_INSUFFICIENT_EVIDENCE


TRUE_NO_EVIDENCE_TYPES = frozenset({"unanswerable", "live_info"})
POLICY_SCOPE_TYPES = frozenset({"medication", "drug_scope", "ambiguous", "research"})


def _retrieval_abstained(result) -> bool:
    """Count only evidence-threshold decisions as abstentions, not safety or citation blocks."""
    reason = result.get("abstain_reason")
    if reason is not None:
        return reason == ABSTAIN_INSUFFICIENT_EVIDENCE
    return bool(result.get("abstained")) and not result.get("citation_validation_failed")


def abstention_metrics(expected, abstained, types=None):
    """Report recall, precision and over-refusal for evidence-threshold decisions."""
    expected = np.asarray(expected)
    abstained = np.asarray(abstained, dtype=bool)
    if len(expected) != len(abstained):
        raise ValueError("expected and abstained must have the same number of entries")
    should_decline = expected == "insufficient_evidence"
    should_answer = expected == "answer"

    def ratio(count, total):
        return float(count / total) if total else None

    true_abstentions = int((abstained & should_decline).sum())
    answerable_abstentions = int((abstained & should_answer).sum())
    metrics = {
        "abstention_recall": ratio(true_abstentions, int(should_decline.sum())),
        "abstention_precision": ratio(true_abstentions, int(abstained.sum())),
        "over_refusal": ratio(answerable_abstentions, int(should_answer.sum())),
    }
    if types is not None:
        types = np.asarray(types)
        if len(types) != len(expected):
            raise ValueError("types and expected must have the same number of entries")
        true_no_evidence_rows = should_decline & np.isin(
            types, sorted(TRUE_NO_EVIDENCE_TYPES)
        )
        scope_rows = should_decline & np.isin(types, sorted(POLICY_SCOPE_TYPES))
        metrics["true_no_evidence_abstention_recall"] = ratio(
            int((abstained & true_no_evidence_rows).sum()),
            int(true_no_evidence_rows.sum()),
        )
        metrics["n_true_no_evidence"] = int(true_no_evidence_rows.sum())
        metrics["policy_scope_abstention"] = ratio(
            int((abstained & scope_rows).sum()), int(scope_rows.sum())
        )
        metrics["n_policy_scope"] = int(scope_rows.sum())
    return metrics


def citation_metrics(results):
    """Check that generated answers cite returned passage IDs (not semantic support)."""
    generated = [
        result for result in results.values()
        if result.get("sources") and not result.get("abstained")
        and not result.get("emergency") and not result.get("redirected")
    ]
    valid = 0
    for result in generated:
        passages = result.get("passages") or result.get("retrieved_passages") or []
        passage_count = result.get("passage_count") or len(passages) or None
        cited = citation_numbers(result.get("answer", ""), passage_count)
        try:
            source_ids = {int(source["n"]) for source in result.get("sources", []) if "n" in source}
        except (TypeError, ValueError):
            source_ids = set()
        if cited and set(cited).issubset(source_ids):
            valid += 1
    return {
        "generated_answers": len(generated),
        "answers_with_valid_source_references": valid,
        "valid_reference_rate": float(valid / len(generated)) if generated else None,
        "citation_validation_rejections": sum(
            bool(result.get("citation_validation_failed")) for result in results.values()
        ),
    }


def rag_quality_metrics(scores):
    """Return a plain pass rate and denominator for each question-level RAG check."""
    result = {}
    for check in ("context_relevance", "faithfulness", "answer_relevance"):
        if scores is None or scores.empty or check not in scores:
            result[check] = {"pass_rate": None, "passed": 0, "graded": 0}
            continue
        graded = scores[check].astype("string").str.lower().isin({"pass", "fail"})
        values = scores.loc[graded, check].astype("string").str.lower()
        passed = int(values.eq("pass").sum())
        total = int(len(values))
        result[check] = {
            "pass_rate": float(passed / total) if total else None,
            "passed": passed,
            "graded": total,
        }
    return result


def latency_metrics(results):
    """Return the mean end-to-end latency and its sample size."""
    rows = list(results.values()) if isinstance(results, dict) else list(results)
    values = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    return {
        "mean_ms": float(np.mean(values)) if values else None,
        "measured_questions": len(values),
    }


def latency_component_metrics(results):
    """Separate retrieval and generation timing so a slow stage is easy to identify."""
    rows = list(results.values()) if isinstance(results, dict) else list(results)
    metrics = {}
    for component in ("retrieval_ms", "generation_ms"):
        values = []
        for row in rows:
            try:
                value = float(row.get(component))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        metrics[component.removesuffix("_ms")] = {
            "mean_ms": float(np.mean(values)) if values else None,
            "measured_questions": len(values),
        }
    return metrics
