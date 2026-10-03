# Schema Validation Report

## PubMedQA (`qiaojin/PubMedQA`, `pqa_artificial`)
- Rows: 211,269
- Columns: ['pubid', 'question', 'context', 'long_answer', 'final_decision']
- Nulls total: 0

## MedQuAD (`lavita/MedQuAD`)
- Rows: 47,441
- Columns: ['document_id', 'document_source', 'document_url', 'category', 'umls_cui', 'umls_semantic_types', 'umls_semantic_group', 'synonyms', 'question_id', 'question_focus', 'question_type', 'question', 'answer']
- Records with answers: 16,407
- Records without answer: 31,034

## MedlinePlus (NLM XML)
- Source: https://medlineplus.gov/xml/mplus_topics_compressed_2026-10-03.zip
- Rows (English only): 1,017
- Columns: ['title', 'url', 'meta_desc', 'summary', 'groups', 'also_called']
