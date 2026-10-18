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

`safety_probes.csv` is a separate regression set of 88 questions for the pre-retrieval safety rules (crisis, swallowed-or-too-much, acute medical, harmful requests, educational decoys, personal-advice redirects). `tests/test_safety_probes.py` requires every live-emergency probe to be caught and every educational probe to reach retrieval. Re-measure it whenever `src/safety.py` changes, and update the counts here if you add rows.

`safety_unseen_probes.csv` measures responses to phrasings that were not used to write the rules. It is a gap check, not a safety certification. On the 2026-10-14 snapshot, the rules escalated 3 of 41 crisis, medical, and ingestion probes, withheld 1 of 9 harmful-request probes, and intercepted 5 of 13 educational probes. These results show that the keyword rules miss many unseen phrasings and also catch some educational questions. `tests/test_safety_unseen.py` pins minimums at those measured values so future changes cannot silently make them worse; the floors are not target performance. Re-measure when changing the rules or adding probes.

`key_facts` provides the labels for key-fact coverage: the judge grades each listed fact against the generated answer (`yes`, `partial` or `no`). Facts are pipe-separated (` | `), and every question labelled `answer` must have at least one.

The threshold is chosen only from dev rows labelled `answer` and from the `insufficient_evidence` rows whose `type` is a genuine no-evidence case (`unanswerable`, `live_info`). The remaining declined rows are scope/policy cases (`medication`, `drug_scope`, `ambiguous`, `research`) whose retrieval scores overlap the answerable ones; they are reported as their own slice so they cannot pull the threshold up and refuse answerable questions. A row whose question is already handled by a safety rule is labelled with that behaviour (`emergency`, `safety_redirect`, `refuse`), never `insufficient_evidence`, because it never reaches the threshold. Safety-rule cases are evaluated separately. There are no gold relevant-passage annotations, so retrieval recall/nDCG must not be claimed from this CSV.
