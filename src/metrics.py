"""Small, deterministic metrics used by the evaluation notebook."""

import math

import numpy as np

from src.generator import citation_numbers
from src.pipeline import ABSTAIN_INSUFFICIENT_EVIDENCE


TRUE_NO_EVIDENCE_TYPES = frozenset({"unanswerable", "live_info"})
POLICY_SCOPE_TYPES = frozenset({"medication", "drug_scope", "ambiguous", "research"})


def wilson_interval(successes, total, z=1.96):
    """Return a 95% Wilson interval, or None when there are no observations."""
    if total is None or total <= 0:
        return None
    successes = min(max(int(successes), 0), int(total))
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


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
        "abstention_recall_ci95": wilson_interval(true_abstentions, int(should_decline.sum())),
        "over_refusal_ci95": wilson_interval(answerable_abstentions, int(should_answer.sum())),
    }
    if types is not None:
        types = np.asarray(types)
        if len(types) != len(expected):
            raise ValueError("types and expected must have the same number of entries")
        scope_rows = should_decline & np.isin(types, sorted(POLICY_SCOPE_TYPES))
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


def coverage_metrics(facts):
    """Report key-fact coverage by fact and by question, with both denominators."""
    if facts is None or len(facts) == 0:
        return {
            "key_fact_coverage": None,
            "key_fact_coverage_question_weighted": None,
            "facts_graded": 0,
            "questions_graded": 0,
        }
    graded = facts.dropna(subset=["score"])
    if graded.empty:
        return {
            "key_fact_coverage": None,
            "key_fact_coverage_question_weighted": None,
            "facts_graded": 0,
            "questions_graded": 0,
        }
    per_question = graded.groupby("id")["score"].mean()
    return {
        "key_fact_coverage": float(graded["score"].mean()),
        "key_fact_coverage_question_weighted": float(per_question.mean()),
        "facts_graded": int(len(graded)),
        "questions_graded": int(per_question.size),
    }


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
