"""Generation: one grounded prompt, one LLM call. There is no fallback to the model's own knowledge."""
import os

from openai import OpenAI

from src import config

SYSTEM_PROMPT = """You are an educational health information assistant. You are not a doctor and you do not give medical advice.

Answer the question using ONLY the numbered evidence passages. Rules:
1. Use only facts stated in the passages. Do not use outside medical knowledge, even if you know the answer, and do not fill gaps from memory.
2. Cite the passages you used, like [1] or [2][3], after the statements they support.
3. If the passages do not answer the question, or answer only part of it, say clearly what the sources do not cover. Do not guess.
4. If the passages disagree or sound uncertain, say so.
5. If the question describes symptoms that could be an emergency (for example chest pain, signs of a stroke, trouble breathing, severe bleeding, overdose or poisoning, thoughts of self-harm), start by telling the person to contact local emergency services or go to the nearest emergency department now.
6. Do not give personal medical advice: no diagnosis, no medicine or dose for a specific person, no advice to start, stop or change a medicine, no interpretation of someone's test results. Explain what the sources say in general and suggest asking a doctor or pharmacist.
7. Do not give instructions that could be used to harm oneself or others.
8. Use plain language and short paragraphs or bullet points (•). No bold, headings or tables."""


def build_messages(question, passages):
    evidence = "\n\n".join(f"[{n}] {p['title']} ({p['source']})\n{p['text']}" for n, p in enumerate(passages, start=1))
    user = (
        f"Evidence passages:\n\n{evidence}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above and cite them as [1], [2], ..."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def make_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=key, base_url=config.LLM_BASE_URL, max_retries=3)   # retries rate limits with backoff


def generate(client, question, passages):
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=build_messages(question, passages),
        max_completion_tokens=config.MAX_TOKENS,
        temperature=0.0,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:   # a reasoning model can spend the whole token budget thinking
        raise RuntimeError(f"The model returned an empty answer (finish_reason={response.choices[0].finish_reason}).")
    return answer
