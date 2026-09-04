"""Tests for lookup planning and parallel execution.

Concurrency and a time budget are the parts most likely to break quietly — a plan that
silently returns nothing, or a slow angle that stalls every lead behind it — so those get
explicit coverage with fake search clients.
"""

import time
from datetime import datetime

import pytest

from agents.planner import SearchQuery, plan_lookups, run_lookups


class FakeSearch:
    def __init__(self, delay=0.0, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on or set()
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        if self.delay:
            time.sleep(self.delay)
        if any(token in query for token in self.fail_on):
            raise RuntimeError("search failed")
        return {"results": [{"content": f"result for {query}", "title": "t", "url": "u"}]}


class TestPlanning:
    def test_role_shapes_the_angles(self):
        sales = {angle for angle, _ in
                 [(q.angle, q) for q in plan_lookups({"company": "A", "role": "Head of Sales"})]}
        marketing = {q.angle for q in plan_lookups({"company": "A", "role": "VP Marketing"})}

        assert sales != marketing
        assert "product_launch" in marketing

    def test_engineering_roles_get_technical_angles(self):
        angles = {q.angle for q in plan_lookups({"company": "A", "role": "VP Engineering"})}
        assert "product_launch" in angles

    def test_unknown_role_falls_back_to_defaults(self):
        angles = {q.angle for q in plan_lookups({"company": "A", "role": "Librarian"})}
        assert angles == {"funding", "product_launch", "hiring"}

    def test_exec_titles_match_even_in_unusual_roles(self):
        """"Chief Vibes Officer" is still a chief — exec angles are the right default."""
        angles = {q.angle for q in plan_lookups({"company": "A", "role": "Chief Vibes Officer"})}
        assert "funding" in angles

    def test_company_name_is_in_every_query(self):
        for q in plan_lookups({"company": "Razorpay", "role": "Head of Sales"}):
            assert "Razorpay" in q.query

    def test_queries_carry_a_recency_signal(self):
        """Regression: keyword-only queries returned a 2021 funding article for a 2026 run,
        even with Tavily's days=30 filter on. The year and "recent news" do real work."""
        year = str(datetime.now().year)
        for q in plan_lookups({"company": "Razorpay", "role": "Head of Sales"}):
            assert "recent news" in q.query
            assert year in q.query

    def test_no_company_means_no_lookups(self):
        assert plan_lookups({"company": "", "role": "Head of Sales"}) == []

    def test_respects_max_queries(self):
        assert len(plan_lookups({"company": "A", "role": "Head of Sales"}, max_queries=2)) == 2

    def test_plan_is_deterministic(self):
        lead = {"company": "A", "role": "Head of Sales"}
        assert plan_lookups(lead) == plan_lookups(lead)


class TestParallelExecution:
    def test_runs_every_query(self):
        search = FakeSearch()
        queries = [SearchQuery("funding", "A funding"), SearchQuery("hiring", "A hiring")]

        results = run_lookups(search, queries)

        assert len(results) == 2
        assert len(search.queries) == 2

    def test_queries_run_concurrently_not_sequentially(self):
        """Three 0.3s lookups must take about 0.3s, not 0.9s — otherwise multi-angle
        research would make every lead three times slower."""
        search = FakeSearch(delay=0.3)
        queries = [SearchQuery(f"a{i}", f"query {i}") for i in range(3)]

        started = time.time()
        run_lookups(search, queries)
        elapsed = time.time() - started

        assert elapsed < 0.7

    def test_one_failing_angle_does_not_lose_the_others(self):
        search = FakeSearch(fail_on={"funding"})
        queries = [SearchQuery("funding", "A funding"), SearchQuery("hiring", "A hiring")]

        results = run_lookups(search, queries)

        assert len(results) == 1
        assert results[0][0].angle == "hiring"

    def test_budget_drops_slow_queries_and_keeps_fast_ones(self):
        search = FakeSearch(delay=1.5)
        queries = [SearchQuery("funding", "A funding")]

        results = run_lookups(search, queries, budget_seconds=0.3)

        assert results == []

    def test_empty_plan_short_circuits(self):
        search = FakeSearch()
        assert run_lookups(search, []) == []
        assert search.queries == []

    def test_results_are_paired_with_their_query(self):
        search = FakeSearch()
        queries = [SearchQuery("funding", "A funding")]

        (query, payload), = run_lookups(search, queries)

        assert query.angle == "funding"
        assert "A funding" in payload["results"][0]["content"]
