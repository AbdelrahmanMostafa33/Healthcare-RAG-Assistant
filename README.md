# Healthcare RAG Assistant

**An evidence-grounded RAG system for answering general health questions from curated healthcare sources.**

Healthcare RAG Assistant is an educational health-information assistant. It answers general questions about diseases, symptoms, causes, prevention, tests and treatments from curated NIH sources, shows which sources it used, and **abstains when the knowledge base does not contain enough relevant information**.

> **Not medical advice.** It is an information assistant, not an AI doctor. It does not diagnose, give personal advice, recommend doses, or replace a doctor or pharmacist.

## Problem

Health information online is large, scattered and uneven. Search engines return pages, not answers. A plain LLM gives fluent answers but cannot show where a claim comes from, and it may state wrong things confidently. For health questions, an answer you cannot trace to a source is hard to trust.

RAG (retrieval-augmented generation) addresses this: the system first searches a curated knowledge base, then asks the LLM to answer **only** from the passages it found, citing them. If nothing relevant is found, it does not ask the LLM at all.

## Scope

| | |
|---|---|
| **In scope** | Diseases and conditions: what they are, their symptoms, causes and risk factors, prevention, tests and diagnosis, and how they are treated in general |
| **Out of scope** | Specific medicines (doses, side effects, interactions, whether to start, stop or combine), diagnosing the user, personal advice, interpreting someone's test results, and live or current information |

**Why medicines are out of scope.** The sources describe how conditions are treated in general, the kinds of medicine used, not specific drugs. Neither MedQuAD's open subset nor MedlinePlus covers drug-specific information (both removed for copyright). So the assistant is scoped to what the sources support, and questions about named drugs are expected to abstain. The evaluation includes such questions to check this.

An assistant that abstains on medicines is safer than one that answers from its own memory: every abstention is an explicit "I don't know", not a confident guess.

## How it works

```
User question
  -> safety rules (src/safety.py), two small keyword checks
        emergency        -> fixed emergency message with emergency numbers   (no retrieval, no LLM)
        personal advice  -> fixed redirect to a doctor or pharmacist         (no retrieval, no LLM)
  -> embed the question                                    (BAAI/bge-m3)
  -> FAISS: the 20 closest chunks                          (exact inner-product search)
  -> cross-encoder rerank, keep the best 5                 (BAAI/bge-reranker-v2-m3)
  -> check evidence: best rerank score >= threshold ?
        no  -> abstain (no LLM call)
        yes -> grounded prompt with numbered passages -> one LLM call   (gpt-oss-120b on Groq)
  -> answer with [1], [2] citations + the sources used + disclaimer
```

| Step | Why it is there |
|---|---|
| Safety rules | An acute emergency or a request for a personal dose or diagnosis should never get a general-information answer. The rules only match when the question describes a real person ("I have...", "my son...") and are tested against educational questions they must not catch ("What are the symptoms of a heart attack?", "What causes colic in babies?"). They are keyword rules, so they miss rephrasings; they are a safety net, not triage. |
| Dense retrieval | Finds chunks about the same thing as the question even when the wording differs. Fast. |
| Rerank | The cross-encoder reads the question and a chunk together, so it judges relevance better than comparing two separate embeddings. Too slow for the whole index, so it only sees the 20 candidates. Its score is also the evidence signal for the next step. |
| Evidence check | Retrieval always returns something. Without a check, the LLM would be asked to answer from irrelevant text. The threshold is chosen on the dev questions (see Evaluation). |
| Grounded prompt | Numbered passages and strict rules (use only these, cite them, say what is missing, no personal advice) make answers checkable. |
| Sources | Only the passages that were in the prompt are returned, with source, title and URL. |

There is no query classifier, no second retrieval pass, no retry with a looser prompt and no fallback to the model's own knowledge.

## Knowledge base

| Source | Content |
|---|---|
| [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | Question/answer pairs written by NIH websites (NIDDK, CDC, NHLBI, ...). Kept with each document's URL and publisher so answers can be cited. |
| [MedlinePlus](https://medlineplus.gov/xml.html) | One plain-language summary per health topic (NLM). |

Documents are cleaned (HTML and NIH site text removed, duplicates dropped) and split into chunks of at most 1000 characters made of whole sentences, with at most one sentence (200 characters) repeated between chunks. The document's question is placed on top of every chunk so it keeps its topic. In the last build this gave 15,994 documents and 31,676 chunks. Chunk size and overlap are reasonable defaults, not tuned values.

**Why a small knowledge base.** The sources were chosen to match the questions, not to be large. PubMedQA (research abstracts with generated questions) and MedRedQA (informal Reddit replies) were left out because they fit consumer questions poorly. This is a judgement about fit; their effect on retrieval was not measured.

**The medication gap.** `lavita/MedQuAD` ships without the answers of its drug, A.D.A.M., herbal and GARD subsets (31,034 rows, removed for copyright), and MedlinePlus drug pages are AHFS content copyrighted by ASHP, so they are not used. The open alternative is FDA labels (DailyMed / openFDA), which are written for clinicians and need their own ingestion step that was not built. Notebook 01 shows the gap in the data; notebook 04 reports medication questions separately.

The index is not stored in the repository. Notebooks 01 and 02 build it.

## Tech stack

Python, pandas, BeautifulSoup, sentence-transformers (embeddings and reranker), FAISS, the Groq API through the OpenAI SDK, FastAPI with Pydantic, pytest, Jupyter.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # with a GPU, install the matching torch build first
cp .env.example .env                 # then add your GROQ_API_KEY
```

1. Build the knowledge base and index by running `notebooks/01_data_and_kb.ipynb`, then `notebooks/02_build_index.ipynb`. A GPU is recommended for the embedding step.
2. Start the API and ask a question:

```bash
uvicorn app:app
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of type 2 diabetes?"}'
```

Interactive docs are at `http://localhost:8000/docs`. Startup loads the models, so it takes a little while. It fails immediately, with a clear message, if `GROQ_API_KEY` is missing or the index has not been built.

3. Run the tests. They use small fakes, so no models, GPU, index or API key are needed:

```bash
pytest
```

## API

`POST /query` with `{"question": "..."}` (3 to 1000 characters) returns:

| Field | Meaning |
|---|---|
| `answer` | The answer with `[n]` citations, the abstention message, or a fixed emergency or redirect message |
| `abstained` | `true` when the sources did not contain enough relevant evidence |
| `emergency` | `true` when the question described an emergency (fixed message, no retrieval, no LLM) |
| `redirected` | `true` when the question asked for personal medical advice (fixed redirect to a doctor, no retrieval, no LLM) |
| `sources` | The passages used: `n` (matches the citation), `source`, `title`, `url`. Empty unless an answer was generated. |
| `disclaimer` | The medical disclaimer |

`GET /health` returns the status, the abstention threshold in use, and whether that threshold has been calibrated. A threshold saved for a different reranker than the one in `src/config.py` is ignored.

## Notebooks

Read them in order. Each states the question it answers. Notebooks 01 and 02 keep the outputs of their last run. Notebooks 03 and 04 are shipped without outputs because their code changed since the last run; run them to see current results.

| Notebook | What it shows |
|---|---|
| `01_data_and_kb` | Scope, sources and why, cleaning, the final knowledge base, and a check of how well it covers the scope |
| `02_build_index` | Chunking, embeddings, the FAISS index, and checks that the index is correct |
| `03_rag_pipeline` | One question walked through retrieval, rerank, evidence check, prompt and LLM call by hand. Then the same thing through `RAGPipeline`, with an emergency, a personal-advice question and an abstention |
| `04_evaluation` | Choosing the abstention threshold, the LLM-only baseline, RAG vs the baseline, failures, and `reports/results.md` |

## Evaluation

**Questions.** `data/eval/questions.csv` has 255 questions, split into dev (168) and test (87). Answerable questions have key facts; the others have an expected behaviour (`insufficient_evidence`, `emergency`, `safety_redirect` or `refuse`).

**Metrics** (all computed by notebook 04, graded by a judge model that is not the generator, every verdict saved in `reports/`):

| Layer | Metrics |
|---|---|
| Generation | Key-fact coverage on questions that should be answered |
| Behaviour | Pass rate for answer, emergency, refuse, safety-redirect and insufficient-evidence questions; abstention recall and precision; over-refusal; latency |

**Abstention threshold.** Chosen on the dev split as the lowest threshold that still abstains on at least 90% of the questions marked `insufficient_evidence`; that answers as many answerable questions as that floor allows. Notebook 04 finds the best rerank score of every dev question first, then chooses and saves the threshold, and only then runs the pipeline, so every table uses the calibrated value. The test split is run once, with the threshold frozen. The saved threshold (3.69) can be reproduced from `reports/rag_dev_answers.csv`; this is checked by a test.

### Results

`reports/results.md` holds the numbers. **They are interim:** the notebook could not be re-run after the latest code changes, so the file was computed offline from the saved judge verdicts, using only questions where the LLM-only baseline and RAG were judged against the same labels and key facts. Headline, on that comparable subset of the dev split:

| | LLM-only | RAG |
|---|---|---|
| Key-fact coverage (47 questions) | 0.983 | 0.726 |

The LLM alone is near the ceiling because most questions concern common health topics, so RAG is **not** better on coverage. What it adds is answers that stay inside the sources, visible citations, and an explicit "not enough information" instead of a confident guess. The saved RAG run also predates the personal-advice redirect (it scored 0/22 on safety-redirect questions by abstaining instead of redirecting), so the current pipeline's safety behaviour is covered by unit tests but has not been evaluated end to end.

## Examples

**Abstention.** Deterministic; no LLM call is made:

```json
{
  "answer": "I couldn't find enough relevant information in my current sources to answer this reliably. Try rephrasing the question, or ask a doctor or pharmacist. If you think this could be an emergency, call your local emergency number or go to the nearest emergency department now.",
  "abstained": true,
  "emergency": false,
  "redirected": false,
  "sources": [],
  "disclaimer": "This is general health information for education only. ..."
}
```

**Emergency and personal advice.** "I have crushing chest pain that spreads to my left arm" returns the fixed emergency message with `"emergency": true`. "How much ibuprofen can I give my 4 year old?" returns a fixed redirect to a doctor or pharmacist with `"redirected": true`. Neither calls retrieval or the LLM. "What are the symptoms of measles in children?" is an educational question and goes through retrieval as normal.

**Grounded answer: the response shape.** The text and sources depend on the knowledge base and the model. Notebook 03 prints real outputs.

```json
{
  "answer": "<answer using only the passages, with [1] [2] citations>",
  "abstained": false,
  "emergency": false,
  "redirected": false,
  "sources": [
    {"n": 1, "source": "MedlinePlus", "title": "<topic>", "url": "https://medlineplus.gov/..."},
    {"n": 2, "source": "MedQuAD (<publisher>)", "title": "<topic>", "url": "https://..."}
  ],
  "disclaimer": "..."
}
```

## Limitations

- **Educational use only.** No diagnosis, no personal advice, no doses. It knows nothing about the user.
- **Safety rules are keywords.** They catch the common phrasings and miss others ("I can't feel my left arm and my chest hurts" is caught; many rewordings will not be). A missed emergency falls through to retrieval, where the prompt still tells the model to point to emergency services. The emergency numbers and crisis lines in `src/safety.py` are for Egypt, the US, the EU and the UK.
- **Knowledge base.** NIH consumer-health content in English. It is a static snapshot: MedQuAD is 2019 data, so some content is dated, and there is no live information. There is no drug source.
- **Retrieval limits the answer.** If the right passage is not retrieved the answer is incomplete, and if the threshold is too strict, answerable questions are refused (about a quarter of them in the saved run).
- **Evaluation.** Key facts were written by the project owner and have not been reviewed by a clinician. Answers are graded by an LLM (a preview model). The threshold is tuned on the dev split (136 questions that reach retrieval), which is a noisy estimate. The safety rules were written while looking at some of the evaluation questions, so numbers on this set are optimistic.
- **Not reproduced end to end here.** The index is generated, not shipped. The encoder and the chunk size were not compared against alternatives, and the saved evaluation predates the current safety rules.

## What I tried and removed

An earlier version was reviewed and simplified. The evidence behind each decision differs, so it is stated:

| Removed | Reason | Evidence |
|---|---|---|
| Query classifier and category routing | Trained on labels from a keyword rule that agreed with hand labels on only about 55% of the 150 evaluation questions. In retrieval its categories were compared with chunk categories using a different vocabulary, so it could not change which chunks came back. It also added a model download and a hard dependency. | Label agreement measured; routing mismatch found by code review |
| BM25 / hybrid retrieval | Switched off by default while the documentation described hybrid search. Nothing in the repository showed a gain. It should come back only if it beats dense + rerank in notebook 04. | No measurement either way |
| Fallbacks that answer without evidence (the model's own knowledge, a looser retry prompt, returning a raw chunk) | They contradict the goal of answering only from sources. Replaced by abstention. | Design decision |
| PubMedQA and MedRedQA in the knowledge base | Poor fit for consumer questions, and they dominated the index by size. | Judgement, not measured |
| Chunks made of the question and nothing else | The earlier chunking produced 219,458 of them (23.2% of all chunks). Now the text is split first and the question added afterwards. | Measured |
| API key rotation, API auth, response cache, extra middleware, a static dashboard | Not useful for this project. Key rotation also made the tests need a real API key even with a fake client. | Review; 11 tests failed without a key |
| Gemini judge (`google-genai`) | Its only recorded run failed ("client has been closed": a throwaway `Client` object is garbage-collected, which closes its connection). The saved results were judged on Groq. One judge, one client. | Run error; library source |
| Medication keyword list and the broad "children / should I / is it safe" redirect rules | They sent educational questions ("symptoms of measles in children", "How can I lower my blood sugar?") to the emergency page or the redirect. Replaced by narrow rules in `src/safety.py`; named drugs now go through retrieval and are abstained on by the threshold. | On the 31 educational questions in `tests/test_safety.py`: 23 intercepted by the old rules, 0 by the new ones (a regression test written partly from the failures found, not an unbiased accuracy estimate) |
| BLEU / ROUGE / BERTScore | Not meaningful for checking facts and groundedness. | Judgement |

## Project structure

```
app.py                  FastAPI service (POST /query, GET /health)
src/
  config.py             settings: paths, models, chunk size, k, threshold
  kb.py                 clean the sources, chunk, embed, build and load the FAISS index
  retriever.py          embed -> FAISS top 20 -> rerank -> top 5
  safety.py             emergency and personal-advice rules that run before retrieval
  generator.py          grounded prompt and the single LLM call
  pipeline.py           RAGPipeline.answer(): safety -> retrieve -> evidence check -> generate -> sources
  evaluation.py         judge, scoring, threshold choice, abstention metrics, report tables
notebooks/              01 data, 02 index, 03 pipeline, 04 evaluation
data/eval/questions.csv the 255 evaluation questions
reports/                results.md, the abstention threshold, and the saved per-question answers and judge verdicts
tests/                  behaviour tests (no models or API key needed)
```

## License

MIT, see `LICENSE`.
