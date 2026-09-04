"""Shared fixtures. No test in this suite may touch a real API.

Every LLM and search client is replaced with a fake, so the suite is free, fast, offline,
and deterministic. That matters here beyond the usual reasons: this project's live models
sit behind a 200k tokens-per-day cap, so a test suite that called them would compete with
the eval harness for the same budget.
"""

import pytest


class FakeChain:
    """Stands in for a LangChain `prompt | llm | parser` chain.

    Returns canned responses in order, and records what it was invoked with so tests can
    assert on the prompt inputs (e.g. that a critique actually reached the rewriter).
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if not self.responses:
            raise AssertionError("FakeChain invoked more times than it had responses")
        return self.responses.pop(0)


@pytest.fixture
def fake_chain():
    return FakeChain


@pytest.fixture
def tavily_dict_response():
    """The shape langchain-tavily actually returns — a dict, not a list.

    Pinned here because mistaking this for a list is precisely the bug that made the
    enricher pass an empty context to the model on every lead.
    """
    return {
        "query": "Razorpay recent news",
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [
            {
                "url": "https://example.com/ray",
                "title": "Razorpay launches RAY",
                "content": "Razorpay launched RAY, a conversational AI account manager, "
                           "with IndusInd Bank.",
                "score": 0.9,
                "published_date": "Wed, 03 Sep 2026 10:00:00 GMT",
                "raw_content": None,
                "id": "1",
            },
            {
                "url": "https://example.com/hiring",
                "title": "Razorpay expands enterprise team",
                "content": "The company opened a Bengaluru enterprise unit this quarter.",
                "score": 0.8,
                "published_date": "Mon, 01 Sep 2026 10:00:00 GMT",
                "raw_content": None,
                "id": "2",
            },
        ],
        "response_time": 1.2,
        "request_id": "abc",
    }


@pytest.fixture
def lead():
    return {
        "name": "Priya Sharma",
        "role": "Head of Sales",
        "company": "Razorpay",
        "email": "priya@razorpay.com",
        "website": "razorpay.com",
    }
