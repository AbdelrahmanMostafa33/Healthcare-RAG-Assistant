"""API handler tests that do not start a local socket or load model weights.

Handlers are called directly rather than through TestClient, because the app's lifespan loads
local model weights and the search index, which none of these tests need. A handler's status code
is read from the Response it receives, and its response model is validated here, as FastAPI does.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError

from app import (
    QUERY_TIMEOUT_DETAIL, QUERY_TIMEOUT_SECONDS, HealthResponse, QueryRequest, QueryResponse, app,
    health, query,
)
from src.deadline import DeadlineExceeded
from src.pipeline import ABSTAIN_MESSAGE, DISCLAIMER
from src.safety import REDIRECT_MESSAGE, REFUSAL_MESSAGE

SOURCE = {"n": 1, "source": "MedlinePlus", "title": "Fever", "url": "https://medlineplus.gov/fever.html"}
EMERGENCY_ANSWER = "**This may describe a medical emergency.**\n\nCall your local emergency number immediately."


class FakePipeline:
    threshold, calibrated = 1.5, True

    def __init__(self, abstain=False, error=None, emergency=False, redirect=False, refuse=False):
        self.abstain, self.error, self.emergency, self.redirect, self.refuse = abstain, error, emergency, redirect, refuse
        self.questions = []
        self.deadlines = []
        self.delay = 0

    def answer(self, question, deadline=None):
        self.questions.append(question)
        self.deadlines.append(deadline)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        flags = {"abstained": False, "emergency": False, "redirected": False, "refused": False}
        if self.emergency:
            return {**flags, "answer": EMERGENCY_ANSWER, "emergency": True, "sources": [], "passages": [], "disclaimer": DISCLAIMER}
        if self.redirect:
            return {**flags, "answer": REDIRECT_MESSAGE, "redirected": True, "sources": [], "passages": [], "disclaimer": DISCLAIMER}
        if self.refuse:
            return {**flags, "answer": REFUSAL_MESSAGE, "refused": True, "sources": [], "passages": [], "disclaimer": DISCLAIMER}
        if self.abstain:
            return {**flags, "answer": ABSTAIN_MESSAGE, "abstained": True,
                    "abstain_reason": "insufficient_evidence", "sources": [], "passages": [],
                    "disclaimer": DISCLAIMER}
        return {**flags, "answer": "Fever is a common symptom [1].", "sources": [SOURCE],
                "passages": [{"text": "internal passage text"}], "disclaimer": DISCLAIMER}


@pytest.fixture
def pipeline():
    instance = FakePipeline()
    app.state.pipeline = instance
    return instance


def request_for(pipeline, startup_error=False):
    state = SimpleNamespace(pipeline=pipeline, startup_error=startup_error)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def get_health(pipeline, startup_error=False):
    """Call the handler the way Starlette does, keeping the status code it sets."""
    response = Response()
    body = HealthResponse.model_validate(health(request_for(pipeline, startup_error), response))
    return response.status_code, body.model_dump()


def post_query(pipeline, payload):
    try:
        body = QueryRequest.model_validate(payload)
    except ValidationError:
        return 422, None
    try:
        result = asyncio.run(query(body, request_for(pipeline)))
    except HTTPException as error:
        return error.status_code, error.detail
    # FastAPI applies this response model after the handler; validate it here without a socket.
    return 200, QueryResponse.model_validate(result).model_dump()


def test_query_returns_answer_sources_and_disclaimer(pipeline):
    status, body = post_query(pipeline, {"question": "  What causes a fever?  "})
    assert status == 200
    assert body == {"answer": "Fever is a common symptom [1].", "abstained": False, "abstain_reason": None,
                    "emergency": False, "redirected": False, "refused": False, "sources": [SOURCE],
                    "disclaimer": DISCLAIMER}
    assert pipeline.questions == ["What causes a fever?"]


def test_an_abstention_has_no_sources_and_reports_why(pipeline):
    pipeline.abstain = True
    _, body = post_query(pipeline, {"question": "What is the Zelvarin-Moss syndrome?"})
    assert body["abstained"] is True and body["emergency"] is False and body["sources"] == []
    assert body["abstain_reason"] == "insufficient_evidence"


def test_an_emergency_has_no_sources_and_is_flagged(pipeline):
    pipeline.emergency = True
    _, body = post_query(pipeline, {"question": "I have crushing chest pain"})
    assert body["emergency"] is True and body["abstained"] is False and body["sources"] == []
    assert "emergency" in body["answer"].lower()


def test_a_personal_advice_question_is_flagged_as_redirected(pipeline):
    pipeline.redirect = True
    _, body = post_query(pipeline, {"question": "How much ibuprofen can I give my son?"})
    assert body["redirected"] is True and body["abstained"] is False
    assert body["emergency"] is False and body["sources"] == []


def test_a_harmful_request_is_flagged_as_refused(pipeline):
    pipeline.refuse = True
    _, body = post_query(pipeline, {"question": "How can I make a poison that cannot be detected?"})
    assert body["refused"] is True and body["sources"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}, {"question": "ab"}, {"question": "x" * 1001}])
def test_invalid_questions_never_reach_the_pipeline(pipeline, payload):
    status, _ = post_query(pipeline, payload)
    assert status == 422 and pipeline.questions == []


def test_a_pipeline_failure_returns_502_without_leaking_details(pipeline):
    pipeline.error = RuntimeError("secret internal detail")
    status, detail = post_query(pipeline, {"question": "What causes a fever?"})
    assert status == 502 and "secret" not in detail


def test_query_returns_504_when_its_overall_budget_expires(pipeline, monkeypatch):
    monkeypatch.setattr("app.QUERY_TIMEOUT_SECONDS", 0.01)
    pipeline.delay = 0.05

    status, detail = post_query(pipeline, {"question": "What causes a fever?"})

    assert status == 504
    assert "took too long" in detail


def test_query_passes_the_remaining_budget_as_a_deadline(pipeline):
    post_query(pipeline, {"question": "What causes a fever?"})

    assert pipeline.deadlines and pipeline.deadlines[0] is not None
    assert 0 < pipeline.deadlines[0] - time.monotonic() <= QUERY_TIMEOUT_SECONDS


def test_a_deadline_raised_by_the_pipeline_returns_504(pipeline):
    pipeline.error = DeadlineExceeded("request deadline passed")

    status, detail = post_query(pipeline, {"question": "What causes a fever?"})

    assert status == 504
    assert detail == QUERY_TIMEOUT_DETAIL


def test_health_reports_the_threshold_and_whether_it_is_calibrated(pipeline):
    assert get_health(pipeline) == (
        200,
        {"status": "ok", "abstain_threshold": 1.5, "threshold_calibrated": True},
    )


def test_health_marks_an_uncalibrated_service_as_degraded(pipeline):
    pipeline.calibrated = False
    assert get_health(pipeline)[1]["status"] == "degraded"


@pytest.mark.parametrize(
    ("startup_error", "status"),
    [(False, "loading"), (True, "error")],
)
def test_health_returns_503_with_a_status_word_while_loading_or_failed(startup_error, status):
    code, body = get_health(None, startup_error=startup_error)

    assert code == 503
    # The page keys off this word, so it must not depend on any message wording.
    assert body["status"] == status
    assert body["abstain_threshold"] is None and body["threshold_calibrated"] is False


def test_openapi_exposes_the_query_route_and_response_contract():
    schema = app.openapi()
    assert "/query" in schema["paths"] and "/health" in schema["paths"]
    assert "QueryResponse" in schema["components"]["schemas"]
    assert schema["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/HealthResponse")
