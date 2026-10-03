# EDA Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Generated:** 2026-10-03 21:47:32

---

## 1. KB composition (n = 258,494)

| Source | Records | Share |
|--------|---------|-------|
| pubmedqa | 211,269 | 81.73% |
| medredqa | 29,801 | 11.53% |
| medquad | 16,407 | 6.35% |
| medlineplus | 1,017 | 0.39% |

PubMedQA holds 81.7% of the KB, so any aggregate statistic is effectively a PubMedQA statistic. Terms and lengths are reported per source.

## 2. Category coverage

| Source | With category | Total | Coverage |
|--------|---------------|-------|----------|
| medlineplus | 0 | 1,017 | 0.0% |
| medquad | 16,407 | 16,407 | 100.0% |
| medredqa | 0 | 29,801 | 0.0% |
| pubmedqa | 0 | 211,269 | 0.0% |

Only MedQuAD carries real labels. The classifier is trained on MedQuAD alone.

## 3. Length statistics (median words)

| Source | Question | Context | Answer |
|--------|----------|---------|--------|
| medlineplus | 4 | 0 | 192 |
| medquad | 7 | 0 | 138 |
| medredqa | 8 | 153 | 47 |
| pubmedqa | 15 | 198 | 34 |

PubMedQA and MedRedQA carry long contexts (153-198 words). MedQuAD and MedlinePlus have no context — the answer is the full unit.

## 4. Classifier training set (n = 12,359)

| Label | Count | Share |
|-------|-------|-------|
| General | 5,905 | 47.78% |
| Symptoms | 2,748 | 22.23% |
| Treatment | 2,442 | 19.76% |
| Diagnosis | 730 | 5.91% |
| Prevention | 534 | 4.32% |

Imbalance ratio: 11.06x. Class weights should be applied during training.

## 5. Length correlation

| Pair | Correlation |
|------|-------------|
| question ↔ answer | -0.077 |
| context ↔ answer  | 0.154 |
| question ↔ context| 0.083 |

No strong correlation. Length cannot leak label information.

## 6. Decisions from this EDA

| Decision | Evidence |
|----------|----------|
| Chunk size 700 / overlap 150 | Median context is ~198 words; splitting by sentence boundaries keeps abstracts intact |
| Train classifier on MedQuAD only | Only source with real labels (12,359 mapped rows) |
| Class-balance the classifier | 11.1x imbalance between largest and smallest class |
| Report terms per source | Global wordcloud would show only PubMedQA vocabulary |

## 7. Figures

1. `01_source_distribution.png`
2. `02_medquad_categories.png`
3. `03_length_distributions.png`
4. `04_answer_len_boxplot.png`
5. `05_top_terms_per_source.png`
6. `06_classifier_train_distribution.png`
7. `07_length_correlation.png`
