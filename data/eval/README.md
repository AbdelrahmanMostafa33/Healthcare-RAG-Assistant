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

`safety_probes.csv` is a separate, held-out set of 88 questions used only to check the pre-retrieval safety rules (crisis, swallowed-or-too-much, acute medical, harmful requests, educational decoys, personal-advice redirects). It is not used to design the rules; `tests/test_safety_probes.py` requires every live-emergency probe to be caught and every educational probe to reach retrieval. Re-measure it whenever `src/safety.py` changes, and update the counts here if you add rows.

`key_facts` is reused twice beyond key-fact coverage: the faithfulness judge (supported claims over total claims, no reference answer) and the key-fact-in-context check, which asks whether the retrieved passages contain each expected fact and is a retrieval proxy rather than a generation score.

The threshold is chosen only from dev rows labelled `answer` and from the `insufficient_evidence` rows whose `type` is a genuine no-evidence case (`unanswerable`, `live_info`). The remaining declined rows are scope/policy cases (`medication`, `drug_scope`, `ambiguous`, `research`) whose retrieval scores overlap the answerable ones; they are reported as their own slice so they cannot pull the threshold up and refuse answerable questions. A row whose question is already handled by a safety rule is labelled with that behaviour (`emergency`, `safety_redirect`, `refuse`), never `insufficient_evidence`, because it never reaches the threshold. Safety-rule cases are evaluated separately. There are no gold relevant-passage annotations, so retrieval recall/nDCG must not be claimed from this CSV.
