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

Notebook 02 sets the evidence cutoff using development questions. Notebook 04 checks whether the system finds useful text, follows the expected response behavior, gives answers supported by its sources, and uses valid citation IDs. LLM-judge scores are estimates, not verified answer accuracy.

The current development/test file contains 168 dev and 87 test questions. The test questions have already been reviewed, so treat their results as exploratory. A separate **9-question final holdout** has source-linked reference answers. Run it after freezing system changes. Its reference-match score is also an LLM estimate, not a clinician review.

The final holdout has not been run yet. Previously saved dev/test reports use the earlier evaluation format; Notebook 04 labels them and can show the saved scores. Re-run a split to create a report in the current format. Provider-based runs are opt-in and send questions, retrieved text, and answers to the selected providers.

A separate retrieval check uses 21 questions, each paired with one expected source page. In the October 22, 2026 run, the page ranked first for **20/21** questions and appeared in the top five for **21/21**. This small check is a retrieval diagnostic, not a broad benchmark or answer-accuracy score.

Run the local tests with:

~~~powershell
python -m pytest -q
~~~

## Safety and limits

Safety checks use keyword rules, not a medical classifier. In a local probe, they escalated **3/41** urgent questions, refused **1/9** harmful requests, and incorrectly intercepted **5/13** educational questions. These results show known gaps; they do not establish safety.

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
data/eval/      Evaluation questions and source references
tests/          Local tests
reports/        Local calibration and evaluation results
~~~

Project code is MIT licensed; see **LICENSE**. Data and model weights have their own licenses.
