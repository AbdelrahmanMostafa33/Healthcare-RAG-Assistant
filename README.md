# Healthcare RAG Assistant

A small retrieval-augmented generation (RAG) assistant that searches health sources, answers from the text it finds, and cites those sources.

> **Education only.** This is not a medical tool. It does not diagnose or recommend treatment.

## Try the app

The web page runs at **/**. The API docs are at **/docs**; the readiness check is at **/health**.

The app needs a local search index and an evidence cutoff before it can answer. If the cutoff is missing or out of date, it abstains.

## How it works

~~~text
Question
  -> safety checks
  -> search for 40 text chunks
  -> rerank and keep up to 5 source pages
  -> check whether the evidence is strong enough
  -> answer with citations, or abstain
~~~

The API and evaluation use the same pipeline. It checks for emergencies, personal-advice requests, and harmful requests before search. It also withholds questions that need current information. The answer model only receives evidence above the cutoff. Invalid citation IDs cause the answer to be withheld.

## Set up

Use Python 3.11 or 3.12. The first download needs internet access and several gigabytes of disk space. A GPU is optional.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Add a Google AI Studio key to **.env** as **GOOGLE_API_KEY**. Add a Groq or Cerebras key only if you plan to run the LLM-judged evaluation. Keep keys private and never commit **.env**.

## Build and run

Run the notebooks in order:

1. **01_data_and_kb.ipynb** downloads and cleans the source material.
2. **02_build_index.ipynb** builds the search index and sets the evidence cutoff.
3. **03_rag_pipeline.ipynb** walks through an example question.
4. **04_evaluation.ipynb** checks retrieval, answers, and safety behavior.

Open Jupyter from the activated virtual environment so it can use the installed packages. A CUDA-enabled PyTorch build can speed up indexing when an NVIDIA GPU is available.

Start the web app:

~~~powershell
uvicorn app:app --reload
~~~

Open **http://127.0.0.1:8000/**. The first load may take time while local search models start.

Example API request:

~~~json
{"question":"What are common symptoms of type 2 diabetes?"}
~~~

## Evaluation

Notebook 02 sets the evidence cutoff using development questions. Notebook 04 checks retrieval, answer behavior, source support, and citations. LLM-judge scores are estimates, not verified answer accuracy.

The questions are split between `data/eval/dev_questions.csv` (168 questions) and `data/eval/test_questions.csv` (87 previously reviewed questions). Use dev while improving the system; use test for exploration only. This setup has no untouched final test set.

LLM-judge runs are off by default. When enabled, questions, retrieved text, and answers are sent to the selected providers. Judge scores are estimates, not verified medical accuracy. Reports are saved in `reports/evaluation/`.
Safety-rule tests use one file in `tests/fixtures/safety_probes.csv`. It combines examples with known expected actions and different phrasings that check how the rules generalize.

Run the local tests with:

~~~powershell
python -m pytest -q
~~~

## Safety and limits

Safety checks use keyword rules, not a medical classifier. On a separate 50-question holdout, urgent cases escalated increased from **0/24** to **2/24**, harmful requests refused increased from **0/10** to **1/10**, and educational questions left alone stayed at **16/16**. These results show known gaps; they do not establish safety.

A valid citation ID only confirms that the cited passage was retrieved. It does not prove the passage supports the answer. Sources may be incomplete or out of date. Do not use this assistant for urgent symptoms, diagnosis, treatment, medication decisions, or personal health information.

## Sources and attribution

- **MedQuAD** is downloaded at the pinned revision in `src/config.py`. It is licensed under [CC BY 4.0](https://github.com/abachaa/MedQuAD/blob/master/LICENSE.txt).
- **MedlinePlus** health-topic summaries come from its [official XML downloads](https://medlineplus.gov/xml.html). See its [reuse guidance](https://medlineplus.gov/about/using/usingcontent/).
- **U.S. Preventive Services Task Force (USPSTF)** recommendations cover colorectal, blood-pressure, and breast cancer screening. Their dates and checksums are saved with the source manifest.

Downloaded data, the processed knowledge base, search index, calibration, and evaluation reports are generated locally and are not committed.

## Project files

~~~text
app.py          Web app and API
frontend/       Web page
src/            Search, safety, answer generation, and evaluation
notebooks/      Data, index, pipeline, and evaluation walkthroughs
data/eval/      Development and test questions
tests/          Local tests and their safety examples
reports/        Local calibration and evaluation results
~~~

Project code is MIT licensed; see **LICENSE**. Data and model weights have their own licenses.
