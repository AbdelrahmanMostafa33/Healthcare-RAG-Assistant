# Evaluation results

Generated 2026-10-08 by `notebooks/04_evaluation.ipynb`. Every number below was computed by that notebook.

## Setup

- Generator `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b` (a different model)
- Embeddings `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3`
- Knowledge base: MedQuAD + MedlinePlus, 31,673 chunks
- Abstention threshold 3.69, calibrated on the dev split
- Dev questions: 168. Test questions: 87 (not evaluated yet)

## Retrieval (dev, questions with key facts)

| Variant | Recall@5 | MRR | questions |
|---|---|---|---|
| FAISS only (top 5) | 0.829 | 0.813 | 88 |
| FAISS + reranker (top 5) | 0.829 | 0.776 | 88 |
| FAISS + reranker, chunks without the question line | 0.807 | 0.817 | 88 |

## LLM-only vs RAG (dev)

The abstention threshold was tuned on this split, so abstention numbers here are optimistic.

| Metric | LLM-only | RAG |
|---|---|---|
| Key-fact coverage | 0.982 | 0.620 |
| Behaviour pass: answer | 68/68 | 64/86 |
| Behaviour pass: emergency | 7/7 | 6/7 |
| Behaviour pass: insufficient_evidence | 12/13 | 47/50 |
| Behaviour pass: refuse | 3/3 | 3/3 |
| Behaviour pass: safety_redirect | 8/9 | 0/22 |
| Abstention recall | - | 0.90 |
| Abstention precision | - | 0.47 |
| Over-refusal (answerable questions abstained on) | - | 0.26 |
| Claims supported by the sources | n/a (no sources) | 0.93 |
| Claims not supported | n/a (no sources) | 0.02 |
| Mean latency (s) | 4.7 | 4.2 |

## Coverage by question type (dev)

Medications are out of scope: the knowledge base has no drug source.

| Type | questions | RAG abstained | key-fact coverage, LLM-only | key-fact coverage, RAG |
|---|---|---|---|---|
| ambiguous | 19 | 0.84 | 0.98 | 0.09 |
| comparative | 7 | 0.57 | - | 0.41 |
| consumer | 33 | 0.06 | 0.99 | 0.83 |
| drug_scope | 7 | 0.86 | - | - |
| lifestyle | 5 | 0.2 | - | 0.48 |
| live_info | 3 | 1.0 | - | - |
| medication | 13 | 0.92 | 0.97 | 0.0 |
| mental_health | 5 | 0.2 | - | 0.78 |
| multi_hop | 9 | 0.33 | - | 0.54 |
| pediatric | 7 | 1.0 | - | - |
| personal_advice | 3 | 0.67 | - | - |
| personal_results | 3 | 1.0 | - | - |
| research | 13 | 0.46 | 0.96 | 0.5 |
| safety | 18 | 0.83 | - | - |
| scope_boundary | 5 | 0.0 | - | 0.83 |
| screening | 5 | 0.2 | - | 0.42 |
| unanswerable | 13 | 1.0 | - | - |

## Limitations

- Key facts were written by the project owner and have not been reviewed by a clinician.
- Most questions are answerable from common knowledge, so the LLM-only baseline scores high on coverage. The value of RAG here is grounding, citations and abstention.
- The dev split has only 13 unanswerable questions, so the abstention threshold is a noisy estimate.
- Claims are judged by an LLM (a preview model). Every verdict is saved in `reports/*.csv`.
- MedQuAD is 2019 data, so some content is dated.
