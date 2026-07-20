"""Tests for fact selection.

    py -m unittest tests.test_tailor

Offline, no API key, no network, no cost -- the model is injected as a stub.

Two properties carry the module and both are tested here rather than trusted:
selection can never emit an ungated bullet, and the prompt prefix is
byte-identical across jobs so the provider's cache actually hits. The second
fails silently and only shows up on a bill, which is exactly the kind of bug
that needs a test rather than a code review.
"""

from __future__ import annotations

import unittest

from jobbuddy import tailor

FACT_A = {
    "fact_id": "umbra.etl",
    "org": "Umbra Financial",
    "role": "Manager, Data Engineering",
    "start": "2023-08", "end": "2024-11",
    "numbers": ["10", "5", "4"],
    "entities": ["PySpark", "ETL"],
    "skills": ["etl", "pyspark"],
    "phrasings": ["Cut data generation from 10 to 5 working days by automating "
                  "4 ETL processes in PySpark"],
    "verified": True,
}

FACT_B = {
    "fact_id": "northwind.retrieval",
    "org": "Northwind Labs",
    "role": "AI Engineer",
    "start": "2024-08", "end": None,
    "numbers": ["12", "340", "90"],
    "entities": ["PyTorch", "Triton"],
    "skills": ["pytorch", "retrieval"],
    "phrasings": ["Built a retrieval pipeline serving 12 million daily requests "
                  "in PyTorch, cutting p99 latency from 340ms to 90ms"],
    "verified": True,
}

UNVERIFIED = dict(FACT_A, fact_id="unverified.thing", verified=False,
                  phrasings=["Did something unconfirmed"])

PROFILE = {
    "years_experience": 4.9,
    "constraints": {"never_claim": ["managed a team", "PhD"]},
    "facts": [FACT_A, FACT_B, UNVERIFIED],
}

JOB_ML = {"title": "Senior ML Engineer", "company": "Acme",
          "seniority": "senior", "jd_text": "You will build retrieval systems."}
JOB_DATA = {"title": "Data Engineer", "company": "Globex",
            "seniority": "mid", "jd_text": "ETL pipelines and governance."}


def stub(selected, headline="", unaddressed=(), ok=True, error=None):
    """A json_chat replacement returning a fixed selection."""
    calls: list[list[dict]] = []

    def _chat(messages, **kwargs):
        calls.append(messages)
        if not ok:
            return {"ok": False, "error": error or "boom"}
        return {"ok": True, "cost_usd": 0.001,
                "data": {"selected": list(selected), "headline": headline,
                         "unaddressed": list(unaddressed)}}

    _chat.calls = calls
    return _chat


class PromptCachePrefixIsStable(unittest.TestCase):
    """A per-job leak into message[0] costs ~50x on every later call."""

    def test_two_different_jobs_produce_identical_prefixes(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1}])
        tailor.tailor(PROFILE, JOB_ML, ["retrieval"], chat=chat)
        tailor.tailor(PROFILE, JOB_DATA, ["etl"], chat=chat)

        first, second = chat.calls[0][0]["content"], chat.calls[1][0]["content"]
        self.assertEqual(first, second)

    def test_the_job_really_does_reach_the_user_message(self):
        """Guards against making the prefix stable by dropping the job."""
        chat = stub([{"fact_id": "umbra.etl", "rank": 1}])
        tailor.tailor(PROFILE, JOB_ML, ["retrieval"], chat=chat)
        self.assertIn("Senior ML Engineer", chat.calls[0][1]["content"])

    def test_reordering_facts_does_not_change_the_prefix(self):
        shuffled = dict(PROFILE, facts=[FACT_B, UNVERIFIED, FACT_A])
        self.assertEqual(tailor.build_prefix(PROFILE), tailor.build_prefix(shuffled))

    def test_unverified_facts_never_reach_the_prompt(self):
        prefix = tailor.build_prefix(PROFILE)
        self.assertNotIn("unverified.thing", prefix)

    def test_the_denylist_reaches_the_prompt(self):
        self.assertIn("managed a team", tailor.build_prefix(PROFILE))


class SelectionCannotEmitUngatedText(unittest.TestCase):
    def test_an_invented_number_falls_back_to_approved_phrasing(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1,
                      "text": "Cut data generation by 90% across 4 ETL processes"}])
        result = tailor.tailor(PROFILE, JOB_DATA, chat=chat)
        self.assertEqual(result["bullets"][0]["text"], FACT_A["phrasings"][0])
        self.assertTrue(result["bullets"][0]["fell_back"])

    def test_a_denylisted_claim_falls_back(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1,
                      "text": "Managed a team of 4 ETL engineers"}])
        result = tailor.tailor(PROFILE, JOB_DATA, chat=chat)
        self.assertEqual(result["bullets"][0]["text"], FACT_A["phrasings"][0])

    def test_a_hallucinated_fact_id_is_reported_not_silently_dropped(self):
        """The clearest available signal the model is ungrounded."""
        chat = stub([{"fact_id": "nonexistent.fact", "rank": 1, "text": "Impressive"},
                     {"fact_id": "umbra.etl", "rank": 2}])
        result = tailor.tailor(PROFILE, JOB_DATA, chat=chat)
        self.assertEqual(result["unknown_fact_ids"], ["nonexistent.fact"])
        # The invented id contributes nothing; the real one is kept. The other
        # employer is then restored, because an employer absent from the
        # selection is a gap in the work history rather than a tailoring choice.
        self.assertNotIn("Impressive", [b["text"] for b in result["bullets"]])
        self.assertIn("umbra.etl", [b["fact_id"] for b in result["bullets"]])

    def test_an_unverified_fact_cannot_be_selected(self):
        chat = stub([{"fact_id": "unverified.thing", "rank": 1}])
        result = tailor.tailor(PROFILE, JOB_DATA, chat=chat)
        self.assertEqual(result["bullets"], [])
        self.assertIn("unverified.thing", result["unknown_fact_ids"])

    def test_a_permitted_rewording_survives(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1,
                      "text": "Automated 4 ETL processes in PySpark"}])
        result = tailor.tailor(PROFILE, JOB_DATA, chat=chat)
        self.assertEqual(result["bullets"][0]["text"],
                         "Automated 4 ETL processes in PySpark")
        self.assertFalse(result["bullets"][0]["fell_back"])


class RankingSurvivesGating(unittest.TestCase):
    def test_output_is_ordered_by_rank_not_by_response_order(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 2},
                     {"fact_id": "northwind.retrieval", "rank": 1}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertEqual([b["fact_id"] for b in result["bullets"]],
                         ["northwind.retrieval", "umbra.etl"])

    def test_a_rejected_bullet_keeps_its_rank_rather_than_sinking(self):
        """Otherwise the renderer cuts it for length, not for relevance."""
        chat = stub([{"fact_id": "umbra.etl", "rank": 1,
                      "text": "Cut generation by 99% at Goldman Sachs"},
                     {"fact_id": "northwind.retrieval", "rank": 2}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertEqual(result["bullets"][0]["fact_id"], "umbra.etl")

    def test_missing_rank_sorts_last_rather_than_crashing(self):
        chat = stub([{"fact_id": "umbra.etl"},
                     {"fact_id": "northwind.retrieval", "rank": 1}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertEqual(result["bullets"][0]["fact_id"], "northwind.retrieval")

    def test_max_bullets_is_enforced_and_the_cut_is_recorded(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1},
                     {"fact_id": "northwind.retrieval", "rank": 2}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat, max_bullets=1)
        self.assertEqual(len(result["bullets"]), 1)
        self.assertEqual(result["dropped_for_length"], 1)


class EveryBulletIsAttributedToTheRightFact(unittest.TestCase):
    """A citation pointing at the wrong fact is worse than none at all.

    It survives every downstream check -- fact_guard already passed the text,
    the report shows a real fact_id -- while quietly claiming the wrong role
    did the work.
    """

    def test_each_bullet_traces_to_the_fact_that_produced_it(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1},
                     {"fact_id": "northwind.retrieval", "rank": 2}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertTrue(result["attribution_aligned"])
        for bullet in result["bullets"]:
            with self.subTest(fact_id=bullet["fact_id"]):
                source = {"umbra.etl": FACT_A,
                          "northwind.retrieval": FACT_B}[bullet["fact_id"]]
                self.assertEqual(bullet["org"], source["org"])
                self.assertIn(bullet["text"], source["phrasings"])

    def test_attribution_holds_when_some_bullets_fall_back(self):
        """The mixed case -- one passes, one is rejected -- is where a naive
        pairing goes wrong, because the two lists advance at different rates."""
        chat = stub([{"fact_id": "umbra.etl", "rank": 1,
                      "text": "Cut costs by 77% at Meta"},
                     {"fact_id": "northwind.retrieval", "rank": 2}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertTrue(result["attribution_aligned"])
        by_id = {b["fact_id"]: b for b in result["bullets"]}
        self.assertEqual(by_id["umbra.etl"]["text"], FACT_A["phrasings"][0])
        self.assertEqual(by_id["northwind.retrieval"]["text"], FACT_B["phrasings"][0])
        self.assertEqual(by_id["umbra.etl"]["org"], "Umbra Financial")
        self.assertEqual(by_id["northwind.retrieval"]["org"], "Northwind Labs")


class FailureIsReportedNotRaised(unittest.TestCase):
    def test_a_failed_call_returns_an_error_record(self):
        result = tailor.tailor(PROFILE, JOB_ML,
                               chat=stub([], ok=False, error="429 rate limited"))
        self.assertFalse(result["ok"])
        self.assertIn("429", result["error"])
        self.assertEqual(result["bullets"], [])

    def test_an_empty_selection_is_not_an_error(self):
        result = tailor.tailor(PROFILE, JOB_ML, chat=stub([]))
        self.assertTrue(result["ok"])
        self.assertEqual(result["bullets"], [])

    def test_unaddressed_requirements_are_passed_through_honestly(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1}],
                    unaddressed=["Kubernetes at scale", "10 years experience"])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertIn("Kubernetes at scale", result["unaddressed"])

    def test_the_guard_summary_travels_with_the_result(self):
        chat = stub([{"fact_id": "umbra.etl", "rank": 1,
                      "text": "Cut costs 80% at Meta"}])
        result = tailor.tailor(PROFILE, JOB_ML, chat=chat)
        self.assertGreaterEqual(result["guard"]["rejected"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
