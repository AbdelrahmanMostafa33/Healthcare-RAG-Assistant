# Evaluation questions

`questions.csv` is a small, hand-curated evaluation set for this educational health RAG project. It is not a clinical validation dataset and has not been reviewed by clinicians.

## Columns

- `id`: stable unique question ID.
- `split`: `dev` for threshold selection and iterative debugging; `test` for a final, frozen evaluation. Do not use test examples to tune rules or thresholds.
- `type`: human-authored scenario group used for slice analysis.
- `category`: broad topic label retained for inspection.
- `expected_behavior`: one of `answer`, `insufficient_evidence`, `emergency`, `safety_redirect`, or `refuse`.
- `question`: the user-facing question.
- `key_facts`: facts to look for in answers to `answer` questions, separated by ` | `. Other behaviors may have blank or explanatory key facts, but those rows are not scored for key-fact coverage.

## Interpretation

The set has 255 questions (168 dev, 87 test). Exact question duplicates are absent, but related conditions and phrasings can appear in both splits; treat test results as a small held-out check, not proof of generalization. Key facts and expected behavior labels were authored by the project owner. The automated judge is a noisy proxy and its verdicts should be inspected alongside failures.

The threshold is chosen only from dev rows labelled `answer` and `insufficient_evidence`. Safety-rule cases are evaluated separately. There are no gold relevant-passage annotations, so retrieval recall/nDCG must not be claimed from this CSV.
