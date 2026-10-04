# EDA Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Generated:** 2026-10-04 04:40:03

---

## 1. KB composition (n = 256,841)

| Source | Records | Share |
|--------|---------|-------|
| pubmedqa | 211,267 | 82.26% |
| medredqa | 29,581 | 11.52% |
| medquad | 14,978 | 5.83% |
| medlineplus | 1,015 | 0.40% |

PubMedQA holds 81.7% of the KB, so any aggregate statistic is effectively a PubMedQA statistic. Terms and lengths are reported per source.

## 2. Category coverage

| Source | With category | Total | Coverage |
|--------|---------------|-------|----------|
| medlineplus | 0 | 1,015 | 0.0% |
| medquad | 14,978 | 14,978 | 100.0% |
| medredqa | 0 | 29,581 | 0.0% |
| pubmedqa | 0 | 211,267 | 0.0% |

Only MedQuAD carries real labels. The classifier is trained on MedQuAD alone.

## 3. Length statistics (median words)

| Source | Question | Context | Answer |
|--------|----------|---------|--------|
| medlineplus | 4 | 0 | 193 |
| medquad | 7 | 0 | 141 |
| medredqa | 8 | 153 | 47 |
| pubmedqa | 15 | 200 | 34 |

PubMedQA and MedRedQA carry long contexts (153-198 words). MedQuAD and MedlinePlus have no context — the answer is the full unit.

## 4. Classifier training set (n = 11,020)

| Label | Count | Share |
|-------|-------|-------|
| General | 5,009 | 45.45% |
| Symptoms | 2,683 | 24.35% |
| Treatment | 2,213 | 20.08% |
| Diagnosis | 687 | 6.23% |
| Prevention | 428 | 3.88% |

Imbalance ratio: 11.70x. Class weights should be applied during training.

## 5. Length correlation

| Pair | Correlation |
|------|-------------|
| question ↔ answer | -0.076 |
| context ↔ answer  | 0.152 |
| question ↔ context| 0.087 |

No strong correlation. Length cannot leak label information.

## 6. Decisions from this EDA

| Decision | Evidence |
|----------|----------|
| Chunk size 1000 / overlap 150 | Median context is ~198 words; 1000 characters keeps a full abstract section in one chunk |
| Train classifier on MedQuAD only | Only source with real labels (12,359 mapped rows) |
| Class-balance the classifier | 11.7x imbalance between largest and smallest class |
| Report terms per source | Global wordcloud would show only PubMedQA vocabulary |

## 7. Figures

1. `01_source_distribution.png`
2. `02_medquad_categories.png`
3. `03_length_distributions.png`
4. `04_answer_len_boxplot.png`
5. `05_top_terms_per_source.png`
6. `06_classifier_train_distribution.png`
7. `07_length_correlation.png`
