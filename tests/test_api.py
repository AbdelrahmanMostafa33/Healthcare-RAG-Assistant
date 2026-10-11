"""API handler tests that do not start a local socket or load model weights."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import QueryRequest, QueryResponse, app, health, query
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
def pipeline():
    instance = FakePipeline()
    app.state.pipeline = instance
    return instance


def request_for(pipeline):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pipeline=pipeline)))


def post_query(pipeline, payload):
    try:
        body = QueryRequest.model_validate(payload)
    except ValidationError:
        return 422, None
    try:
        result = query(body, request_for(pipeline))
    except HTTPException as error:
        return error.status_code, error.detail
    # FastAPI applies this response model after the handler; validate it here without a socket.
    return 200, QueryResponse.model_validate(result).model_dump()


def test_query_returns_answer_sources_and_disclaimer(pipeline):
    status, body = post_query(pipeline, {"question": "  What causes a fever?  "})
    assert status == 200
    assert body == {"answer": "Fever is a common symptom [1].", "abstained": False, "emergency": False,
                    "redirected": False, "sources": [SOURCE], "disclaimer": DISCLAIMER}
    assert pipeline.questions == ["What causes a fever?"]


def test_an_abstention_has_no_sources(pipeline):
    pipeline.abstain = True
    _, body = post_query(pipeline, {"question": "What is the Zelvarin-Moss syndrome?"})
    assert body["abstained"] is True and body["emergency"] is False and body["sources"] == []


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


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}, {"question": "ab"}, {"question": "x" * 1001}])
def test_invalid_questions_never_reach_the_pipeline(pipeline, payload):
    status, _ = post_query(pipeline, payload)
    assert status == 422 and pipeline.questions == []


def test_a_pipeline_failure_returns_502_without_leaking_details(pipeline):
    pipeline.error = RuntimeError("secret internal detail")
    status, detail = post_query(pipeline, {"question": "What causes a fever?"})
    assert status == 502 and "secret" not in detail


def test_health_reports_the_threshold_and_whether_it_is_calibrated(pipeline):
    assert health(request_for(pipeline)) == {"status": "ok", "abstain_threshold": 1.5, "threshold_calibrated": True}


def test_health_marks_an_uncalibrated_service_as_degraded(pipeline):
    pipeline.calibrated = False
    assert health(request_for(pipeline))["status"] == "degraded"


def test_openapi_exposes_the_query_route_and_response_contract():
    schema = app.openapi()
    assert "/query" in schema["paths"] and "/health" in schema["paths"]
    assert "QueryResponse" in schema["components"]["schemas"]
