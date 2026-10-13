"""Reported metrics: turn saved per-row verdicts into the numbers in the report.

`evaluation.py` runs the batch and calls the judge; this module only computes. Nothing here makes a
provider call or reads a file, so the arithmetic can be reviewed and tested on its own, without
reading the judge plumbing first.
"""
import math
import re

import numpy as np
import pandas as pd

from src.pipeline import ABSTAIN_INSUFFICIENT_EVIDENCE

TRUE_NO_EVIDENCE_TYPES = frozenset({"unanswerable", "live_info"})
POLICY_SCOPE_TYPES = frozenset({"medication", "drug_scope", "ambiguous", "research"})

def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a proportion, or None when there is nothing to estimate.

    Wilson rather than the normal approximation: these samples are small (16 no-evidence dev
    questions) and the rates sit near 1.0, where the normal interval can exceed 1.0.
    """
    if total is None or total <= 0:
        return None
    successes = min(max(int(successes), 0), int(total))
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _retrieval_abstained(result) -> bool:
    """True only when the evidence threshold blocked generation.

    A citation-validation rejection is not an abstention: the sources did contain an answer, the
    generated text just failed the structural check. Counting it as one turned a single rejected
    answer into over_refusal = 1.0, and a question answered by the safety rules never reaches the
    threshold at all.
    """
    reason = result.get("abstain_reason")
    if reason is not None:
        return reason == ABSTAIN_INSUFFICIENT_EVIDENCE
    return bool(result.get("abstained")) and not result.get("citation_validation_failed")


def abstention_metrics(expected, abstained, types=None):
    """Measure threshold-based abstention, separate from emergency and advice rules.

    ``abstained`` should be true only when the evidence threshold blocked generation.
    Hard-coded emergency responses and personal-advice redirects are evaluated separately
    through their expected-behaviour pass rates.

    recall: share of insufficient-evidence questions that were abstained on.
    precision: share of threshold abstentions that were truly insufficient-evidence questions.
    over_refusal: share of answerable questions that were threshold-abstained.
    policy_scope_abstention: share of scope/policy questions that were abstained on, reported
        separately because those rows are not what the threshold is calibrated against.
    """
    expected = np.asarray(expected)
    abstained = np.asarray(abstained, dtype=bool)
    if len(expected) != len(abstained):
        raise ValueError("expected and abstained must have the same number of entries")
    should_decline = expected == "insufficient_evidence"
    should_answer = expected == "answer"

    def ratio(count, total):
        return float(count / total) if total else None

    metrics = {
        "abstention_recall": ratio((abstained & should_decline).sum(), should_decline.sum()),
        "abstention_precision": ratio((abstained & should_decline).sum(), abstained.sum()),
        "over_refusal": ratio((abstained & should_answer).sum(), should_answer.sum()),
        "abstention_recall_ci95": wilson_interval((abstained & should_decline).sum(), should_decline.sum()),
        "over_refusal_ci95": wilson_interval((abstained & should_answer).sum(), should_answer.sum()),
    }
    if types is not None:
        types = np.asarray(types)
        if len(types) != len(expected):
            raise ValueError("types and expected must have the same number of entries")
        scope = should_decline & np.isin(types, sorted(POLICY_SCOPE_TYPES))
        metrics["policy_scope_abstention"] = ratio((abstained & scope).sum(), scope.sum())
        metrics["n_policy_scope"] = int(scope.sum())
    return metrics


def grounding_metrics(faithfulness, in_context):
    """Mean faithfulness and in-context rate over the rows each judge actually graded."""
    graded_faith = faithfulness.loc[faithfulness["status"] == "ok", "faithfulness"].dropna()
    graded_context = in_context.loc[in_context["status"] == "ok", "in_context_rate"].dropna()
    return {
        "faithfulness": float(graded_faith.mean()) if len(graded_faith) else None,
        "faithfulness_answers": int(len(graded_faith)),
        "key_fact_in_context": float(graded_context.mean()) if len(graded_context) else None,
        "key_fact_in_context_questions": int(len(graded_context)),
        "faithfulness_errors": int((faithfulness["status"] == "error").sum()),
        "key_fact_in_context_errors": int((in_context["status"] == "error").sum()),
    }


def citation_metrics(results):
    """Deterministic citation-reference metrics for RAG results (not semantic entailment)."""
    import re

    generated = [r for r in results.values() if r.get("sources") and not r.get("abstained")
                 and not r.get("emergency") and not r.get("redirected")]
    valid = 0
    for result in generated:
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", result.get("answer", ""))]
        source_ids = {int(source["n"]) for source in result.get("sources", []) if "n" in source}
        if cited and set(cited).issubset(source_ids):
            valid += 1
    return {
        "generated_answers": len(generated),
        "answers_with_valid_source_references": valid,
        "valid_reference_rate": float(valid / len(generated)) if generated else None,
        "citation_validation_rejections": int(sum(bool(r.get("citation_validation_failed")) for r in results.values())),
    }


def summarize(facts, behavior, latency_ms):
    """Key-fact coverage, coverage by type, behaviour pass counts and mean latency for one system."""
    passed = behavior.groupby("expected")["ok"].agg(["sum", "count"])
    return {
        "key_fact_coverage": float(facts["score"].mean()) if len(facts) else None,
        "coverage_by_type": facts.groupby("type")["score"].mean().to_dict(),
        "behavior": {name: f"{int(row['sum'])}/{int(row['count'])}" for name, row in passed.iterrows()},
        "latency_s": float(latency_ms) / 1000,
    }


# --- report tables ---------------------------------------------------------

def _fmt(value, digits=2):
    return "-" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _fmt_interval(bounds, digits=2):
    if not bounds:
        return "-"
    low, high = bounds
    return f"{low:.{digits}f}-{high:.{digits}f}"


def comparison_table(baseline, rag, abstention):
    """LLM-only vs RAG, from two summarize() results and the RAG abstention_metrics()."""
    rows = {"Key-fact coverage": (_fmt(baseline["key_fact_coverage"], 3), _fmt(rag["key_fact_coverage"], 3))}
    for name in ["answer", "emergency", "insufficient_evidence", "refuse", "safety_redirect"]:
        rows[f"Behaviour pass: {name}"] = (baseline["behavior"].get(name, "-"), rag["behavior"].get(name, "-"))
    rows["Abstention recall"] = ("-", _fmt(abstention["abstention_recall"]))
    rows["Abstention recall (95% CI)"] = ("-", _fmt_interval(abstention.get("abstention_recall_ci95")))
    rows["Abstention precision"] = ("-", _fmt(abstention["abstention_precision"]))
    rows["Over-refusal (answerable questions not answered)"] = ("-", _fmt(abstention["over_refusal"]))
    rows["Over-refusal (95% CI)"] = ("-", _fmt_interval(abstention.get("over_refusal_ci95")))
    rows["Abstention on scope/policy questions"] = ("-", _fmt(abstention.get("policy_scope_abstention")))
    rows["Mean latency (s)"] = (_fmt(baseline["latency_s"], 1), _fmt(rag["latency_s"], 1))
    return pd.DataFrame(rows, index=["LLM-only", "RAG"]).T


def grounding_table(metrics):
    """Faithfulness and key-fact-in-context as their own table, so they are never folded into
    key-fact coverage or the abstention numbers."""
    rows = {
        "Faithfulness (supported claims / claims)": _fmt(metrics.get("faithfulness"), 3),
        "Answers judged for faithfulness": str(metrics.get("faithfulness_answers", 0)),
        "Key facts present in retrieved context": _fmt(metrics.get("key_fact_in_context"), 3),
        "Answerable questions judged for context": str(metrics.get("key_fact_in_context_questions", 0)),
        "Faithfulness judge errors": str(metrics.get("faithfulness_errors", 0)),
        "In-context judge errors": str(metrics.get("key_fact_in_context_errors", 0)),
    }
    return pd.DataFrame(rows, index=["RAG"]).T


def md_table(df, first_column=""):
    header = [first_column] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in df.astype(object).iterrows():
        lines.append("| " + " | ".join([str(index)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)
