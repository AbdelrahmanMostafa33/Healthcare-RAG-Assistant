# Evaluation results

**Status:** only the **LLM-only baseline** has been measured. The RAG pipeline has **not** been evaluated.
The repository ships without the knowledge-base index, the models and an API key, so nothing here was run against the final system.
Run `notebooks/04_evaluation.ipynb` after building the index (notebooks 01 and 02). It overwrites this file with numbers computed from the real run:
retrieval (Recall@5 and MRR, reranker vs no reranker, chunks with vs without the question line), the calibrated abstention threshold, claim support,
abstention recall and precision, over-refusal, coverage by question type, and latency.

## LLM-only baseline (dev split, 100 questions)

Generator `openai/gpt-oss-120b` with a plain "helpful assistant" prompt and no retrieval. Judge `qwen/qwen3.8-27b` (a different model).
Verdicts are saved in `reports/baseline_llm_only_dev*.csv`. 69 questions have key facts (171 facts in total).

| Metric | LLM-only (measured) | RAG |
|---|---|---|
| Key-fact coverage | 0.982 | not run yet |
| Behaviour pass: answer | 68/68 | not run yet |
| Behaviour pass: emergency | 7/7 | not run yet |
| Behaviour pass: insufficient_evidence | 12/13 | not run yet |
| Behaviour pass: refuse | 3/3 | not run yet |
| Behaviour pass: safety_redirect | 8/9 | not run yet |
| Mean latency (s) | 4.7 | not run yet |

| Question type | LLM-only key-fact coverage |
|---|---|
| ambiguous | 0.978 |
| consumer | 0.990 |
| medication | 0.970 |
| research | 0.964 |

## How to read the baseline

- Coverage is already very high: these questions are mostly answerable from common knowledge. RAG is **not** expected to beat the baseline on coverage. What it should add is answers that stay inside the sources, citations, and an explicit "not enough information" instead of a confident guess.
- The baseline has no abstention mechanism and no sources, so those rows are empty for it by design.
- The one failed `insufficient_evidence` check and the one failed `safety_redirect` check are saved in `reports/baseline_llm_only_dev_behavior.csv` with the judge's reason.
- The 20 medication questions ask mostly about one named drug. The knowledge base has no drug source, so RAG is expected to abstain on many of them. The LLM-only baseline answers them from its own memory, which is exactly what this project does not allow. The RAG results will report them separately.

## Limitations

- Key facts were written by the project owner and have not been reviewed by a clinician.
- The dev split has only 13 unanswerable questions, so an abstention threshold calibrated on it will be a noisy estimate.
- Answers are judged by an LLM (a preview model). Every verdict is saved so it can be audited.
- The question set was not extended with knowledge-base-specific questions, because that needs the real knowledge base to verify the expected facts.
