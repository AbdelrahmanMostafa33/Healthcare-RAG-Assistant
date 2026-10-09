"""Generation: one grounded prompt, one LLM call. There is no fallback to the model's own knowledge.

Transient generator 429s from notebook regeneration loops are retried a bounded number of times via
the same helpers used in ``src/evaluation``. Tokens-per-day exhaustion is raised immediately and
clearly instead of being retried forever.
"""
import os

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src import config
from src.evaluation import call_with_transient_rate_limit_retry

SYSTEM_PROMPT = """You are an educational health information assistant. You are not a doctor and you do not give medical advice.

Answer the question using ONLY the numbered evidence passages. Do not use outside medical knowledge, even if you know the answer, and do not fill gaps from memory.

Structure the answer with short bold section titles when the passages contain information for them. For a condition, the usual sections are:

**Overview** - what it is, in one or two sentences.
**Symptoms** - what people notice.
**Causes** - why it happens or who is at risk.
**Diagnosis** - how doctors test for it.
**Treatment** - how it is usually managed.
**Prevention** - how to reduce the risk.
**When to see a doctor** - warning signs to act on.

Only include a section when the passages actually have information for it. Do not invent sections and do not pad empty ones. If the question is narrow (for example "what causes X?"), answer it directly and skip the overview.

Rules:
1. Cite the passages you used, like [1] or [2][3], after the statements they support.
2. If the passages answer only part of the question, say clearly what they do not cover.
3. If the passages do not answer the question at all, say so plainly instead of guessing.
4. If the passages disagree or sound uncertain, say so.
5. Do not give personal medical advice — no medicine or dose for a specific person, and no advice about starting, stopping or combining medicines. Medicines are outside this assistant's scope. Explain what the sources say in general and suggest asking a doctor or pharmacist.
6. If the question describes symptoms that could be an emergency (for example chest pain, signs of a stroke, trouble breathing, severe bleeding, overdose or poisoning, thoughts of self-harm), start by telling the person to contact local emergency services now.
7. Do not give instructions that could be used to harm oneself or others.
8. Use plain language. Short paragraphs or bullet points (•) under each section. No tables.
9. Cite using exactly the form [1] or [2][3] — never use other citation formats."""


def make_client(max_retries=3):
    """Groq through the OpenAI SDK. The SDK retries rate limits and server errors with backoff.

    Only one key is used. If GROQ_API_KEY holds a comma-separated list (the old rotation format),
    the first non-empty key is used and the rest are ignored: there is no rotation.
    """
    keys = [key.strip() for key in os.getenv("GROQ_API_KEY", "").split(",") if key.strip()]
    if not keys:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=keys[0], base_url=config.LLM_BASE_URL, max_retries=max_retries, timeout=60)


def build_messages(question: str, passages: list[dict]) -> list[ChatCompletionMessageParam]:
    evidence = "\n\n".join(f"[{n}] {p['title']} ({p['source']})\n{p['text']}" for n, p in enumerate(passages, start=1))
    user = (
        f"Evidence passages:\n\n{evidence}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above and cite them as [1], [2], ..."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(client: OpenAI, question: str, passages: list[dict]) -> str:
    """One LLM call at temperature 0. Raises instead of returning a blank answer.

    Transient generator 429s are retried a bounded number of times. Tokens-per-day exhaustion is
    raised immediately and clearly rather than retried forever (that is the failure notebook 4 now
    hits during the regenerated LLM-only baseline cell).
    """
    return call_with_transient_rate_limit_retry(
        client,
        lambda: _generate_once(client, question, passages),
        max_transient_retries=5,
    )


def _generate_once(client: OpenAI, question: str, passages: list[dict]) -> str:
    """The actual one-shot LLM call. Separated so the retry wrapper can call it without retrying
    the surrounding control flow."""
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=build_messages(question, passages),
        max_completion_tokens=config.MAX_TOKENS,
        temperature=0.0,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:  # a reasoning model can spend the whole token budget thinking
        raise RuntimeError(f"The model returned an empty answer (finish_reason={response.choices[0].finish_reason}).")
    return answer
