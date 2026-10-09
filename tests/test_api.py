import pytest
from fastapi.testclient import TestClient

from app import app
from src.pipeline import ABSTAIN_MESSAGE, DISCLAIMER
from src.safety import REDIRECT_MESSAGE

SOURCE = {"n": 1, "source": "MedlinePlus", "title": "Fever", "url": "https://medlineplus.gov/fever.html"}
EMERGENCY_ANSWER = "**This may describe a medical emergency.**\n\nCall your local emergency number immediately."


class FakePipeline:
    threshold, calibrated = 1.5, True

    def __init__(self, abstain=False, error=None, emergency=False, redirect=False):
        self.abstain, self.error, self.emergency, self.redirect, self.questions = abstain, error, emergency, redirect, []

    def answer(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        flags = {"abstained": False, "emergency": False, "redirected": False}
        if self.emergency:
            return {**flags, "answer": EMERGENCY_ANSWER, "emergency": True, "sources": [], "passages": [], "disclaimer": DISCLAIMER}
        if self.redirect:
            return {**flags, "answer": REDIRECT_MESSAGE, "redirected": True, "sources": [], "passages": [], "disclaimer": DISCLAIMER}
        if self.abstain:
            return {**flags, "answer": ABSTAIN_MESSAGE, "abstained": True, "sources": [], "passages": [], "disclaimer": DISCLAIMER}
        return {**flags, "answer": "Fever is a common symptom [1].", "sources": [SOURCE],
                "passages": [{"text": "internal passage text"}], "disclaimer": DISCLAIMER}


@pytest.fixture
def client():
    # no `with`, so the startup that loads the real models is skipped
    return TestClient(app)


def use(pipeline):
    app.state.pipeline = pipeline
    return pipeline


def test_query_returns_answer_sources_and_disclaimer(client):
    pipeline = use(FakePipeline())
    response = client.post("/query", json={"question": "  What causes a fever?  "})
    assert response.status_code == 200
    assert response.json() == {"answer": "Fever is a common symptom [1].", "abstained": False, "emergency": False,
                               "redirected": False, "sources": [SOURCE], "disclaimer": DISCLAIMER}   # no internal passages
    assert pipeline.questions == ["What causes a fever?"]


def test_an_abstention_has_no_sources(client):
    use(FakePipeline(abstain=True))
    body = client.post("/query", json={"question": "What is the Zelvarin-Moss syndrome?"}).json()
    assert body["abstained"] is True and body["emergency"] is False and body["sources"] == []


def test_an_emergency_has_no_sources_and_is_flagged(client):
    use(FakePipeline(emergency=True))
    body = client.post("/query", json={"question": "I have crushing chest pain"}).json()
    assert body["emergency"] is True
    assert body["abstained"] is False
    assert body["sources"] == []
    assert "emergency" in body["answer"].lower()


def test_a_personal_advice_question_is_flagged_as_redirected(client):
    use(FakePipeline(redirect=True))
    body = client.post("/query", json={"question": "How much ibuprofen can I give my son?"}).json()
    assert body["redirected"] is True and body["abstained"] is False and body["emergency"] is False and body["sources"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}, {"question": "ab"}, {"question": "x" * 1001}])
def test_invalid_questions_never_reach_the_pipeline(client, payload):
    pipeline = use(FakePipeline())
    assert client.post("/query", json=payload).status_code == 422
    assert pipeline.questions == []


def test_a_pipeline_failure_returns_502_without_leaking_details(client):
    use(FakePipeline(error=RuntimeError("secret internal detail")))
    response = client.post("/query", json={"question": "What causes a fever?"})
    assert response.status_code == 502 and "secret" not in response.text


def test_health_reports_the_threshold_and_whether_it_is_calibrated(client):
    use(FakePipeline())
    assert client.get("/health").json() == {"status": "ok", "abstain_threshold": 1.5, "threshold_calibrated": True}