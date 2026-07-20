"""Tests for the resume import and the mechanical verification gate.

    py -m unittest tests.test_profile_import

Offline, no API key, no network, no cost. The model is injected as a stub so
the extraction path is exercised without spending anything.

The cases that matter are the ones where a model fabricates something and the
verifier has to notice -- and, just as importantly, the ones where the PDF
extractor mangles a true span and the verifier must NOT raise a false alarm. A
verifier that cries wolf gets bypassed, and a bypassed verifier is worse than
none at all.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jobbuddy import import_resume, verify_profile

# Stands in for text pypdf pulled out of a one-page resume, with the artefacts
# a real extraction carries: a mid-phrase line break, a smart apostrophe and a
# bullet glyph.
# A fictional person. This repo is public, so no fixture reconstructs a real
# CV -- an earlier fixture leaked a real phone number and had to be scrubbed
# after the fact, which is a worse way to learn the rule.
RESUME_TEXT = """Alex Tan
Singapore | AI Engineer

Northwind Labs — AI Engineer, Aug 2024 – Present
• Built a retrieval pipeline serving 12 million daily requests using PyTorch
  and Triton, cutting p99 latency from 340ms to 90ms

Umbra Financial — Manager, Data Engineering, Aug 2023 – Nov 2024
• Cut data generation from 10 to 5 working days by automating 4 ETL
  processes in PySpark

EDUCATION
Eastvale Institute — MSc (Artificial Intelligence), 2023
"""

TRUE_FACT = {
    "fact_id": "umbra.etl",
    "org": "Umbra Financial",
    "role": "Manager, Data Engineering",
    "start": "2023-08", "end": "2024-11",
    "source_span": "Cut data generation from 10 to 5 working days by automating 4 ETL processes in PySpark",
    "numbers": ["10", "5", "4"],
    "entities": ["PySpark", "ETL"],
    "skills": ["etl", "pyspark"],
}


def stub_chat(payload, ok=True, error=None):
    """A json_chat replacement returning a fixed payload."""
    def _chat(messages, **kwargs):
        if not ok:
            return {"ok": False, "error": error or "boom"}
        return {"ok": True, "data": payload, "repaired": False}
    return _chat


class NormalisationForgivesLayoutNotContent(unittest.TestCase):
    def test_line_break_inside_a_span_still_matches(self):
        """The resume wraps 'ETL\\n  processes'; the model returns one line.

        This is the false-alarm case. Getting it wrong would flag most true
        facts and train the user to skim the review list.
        """
        problems = verify_profile.check_fact(
            TRUE_FACT, verify_profile.normalise(RESUME_TEXT))
        self.assertEqual(problems, [], problems)

    def test_a_pdf_space_before_a_hyphen_does_not_cause_a_false_rejection(self):
        """pypdf extracts "end-to-end" as "end -to-end".

        Four true facts were rejected as paraphrases over this, and the
        transcription had been correct every time -- the verifier was comparing
        against mangled extraction output and blaming the model. The cost was
        not just the four: it was believing the extractor paraphrases, which is
        the opposite of what it does.
        """
        # Three variants of the same artefact, all seen in one real PDF:
        # space before the hyphen, spaces on both sides, and space before a
        # full stop.
        resume = ("Built the end -to-end RAG pipeline with LLM - generated "
                  "output . Live and in use .")
        fact = dict(TRUE_FACT,
                    source_span="Built the end-to-end RAG pipeline with "
                                "LLM-generated output. Live and in use.",
                    numbers=[], entities=[], org="", role="",
                    start=None, end=None)
        problems = verify_profile.check_fact(
            fact, verify_profile.normalise(resume))
        self.assertEqual(problems, [], problems)

    def test_a_genuine_spaced_dash_between_numbers_is_left_alone(self):
        """The fix must not weld "10 - 5" into "10-5", so it anchors on
        letters rather than on word characters."""
        self.assertIn("10 - 5", verify_profile.normalise("cut 10 - 5 days"))
        self.assertIn("2021 - 2024", verify_profile.normalise("MITB 2021 - 2024"))

    def test_smart_punctuation_does_not_break_matching(self):
        fact = dict(TRUE_FACT,
                    source_span="Built a retrieval pipeline serving 12 million daily requests",
                    numbers=["12"], entities=[], org="Northwind Labs", role="AI Engineer",
                    start="2024-08", end=None)
        problems = verify_profile.check_fact(
            fact, verify_profile.normalise(RESUME_TEXT))
        self.assertEqual(problems, [], problems)


class FabricationIsCaught(unittest.TestCase):
    def setUp(self):
        self.resume = verify_profile.normalise(RESUME_TEXT)

    def test_paraphrased_span_is_flagged(self):
        """The commonest failure: the model tidies the wording."""
        fact = dict(TRUE_FACT,
                    source_span="Reduced data generation time from ten days to five "
                                "by automating four ETL processes")
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertTrue(any("not literal text" in p for p in problems), problems)

    def test_invented_number_is_flagged(self):
        fact = dict(TRUE_FACT, numbers=["10", "5", "4", "60"])
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertTrue(any("'60'" in p for p in problems), problems)

    def test_entity_never_in_the_resume_is_flagged(self):
        fact = dict(TRUE_FACT, entities=["PySpark", "Kubernetes"])
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertTrue(any("Kubernetes" in p for p in problems), problems)

    def test_employer_never_worked_at_is_flagged(self):
        fact = dict(TRUE_FACT, org="Goldman Sachs")
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertTrue(any("Goldman" in p for p in problems), problems)

    def test_wrong_year_is_flagged(self):
        fact = dict(TRUE_FACT, start="2019-08")
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertTrue(any("2019" in p for p in problems), problems)

    def test_missing_span_is_flagged_and_stops_there(self):
        fact = dict(TRUE_FACT, source_span="")
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertEqual(len(problems), 1)

    def test_span_too_short_to_be_distinctive_is_rejected(self):
        """A three-word span matches by accident, which is not verification."""
        fact = dict(TRUE_FACT, source_span="ETL", numbers=[], entities=[])
        problems = verify_profile.check_fact(fact, self.resume)
        self.assertTrue(problems)


class AutoVerifySplitsTheWork(unittest.TestCase):
    def test_true_facts_pass_and_fabricated_ones_land_in_review(self):
        draft = {"facts": [
            dict(TRUE_FACT),
            dict(TRUE_FACT, fact_id="invented", numbers=["10", "5", "4", "97"]),
        ]}
        out = verify_profile.auto_verify(draft, RESUME_TEXT)
        self.assertTrue(out["facts"][0]["verified"])
        self.assertFalse(out["facts"][1]["verified"])

        summary = verify_profile.summarise(out)
        self.assertEqual(summary["auto_verified"], 1)
        self.assertEqual(summary["needs_review"], 1)
        self.assertEqual(summary["review"][0]["fact_id"], "invented")

    def test_auto_verify_does_not_mutate_the_input(self):
        draft = {"facts": [dict(TRUE_FACT)]}
        verify_profile.auto_verify(draft, RESUME_TEXT)
        self.assertFalse(draft["facts"][0].get("verified"))

    def test_verification_record_states_what_it_does_not_prove(self):
        """The honesty of this record is the whole point of the module."""
        out = verify_profile.auto_verify({"facts": [dict(TRUE_FACT)]}, RESUME_TEXT)
        record = out["facts"][0]["verification"]
        self.assertIn("does_not_prove", record)


class PromoteRefusesDoubtfulProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "master_profile.json"

    def test_promote_refuses_while_anything_needs_review(self):
        draft = verify_profile.auto_verify(
            {"facts": [dict(TRUE_FACT), dict(TRUE_FACT, fact_id="bad", numbers=["99"])]},
            RESUME_TEXT)
        with self.assertRaises(ValueError) as caught:
            verify_profile.promote(draft, self.tmp)
        self.assertIn("bad", str(caught.exception))
        self.assertFalse(self.tmp.exists())

    def test_allow_unverified_drops_rather_than_promotes(self):
        """A smaller true profile, never a larger doubtful one."""
        draft = verify_profile.auto_verify(
            {"facts": [dict(TRUE_FACT), dict(TRUE_FACT, fact_id="bad", numbers=["99"])]},
            RESUME_TEXT)
        path = verify_profile.promote(draft, self.tmp, allow_unverified=True)
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(written["facts"]), 1)
        self.assertEqual(written["_dropped_unverified"], 1)
        self.assertTrue(all(f["verified"] for f in written["facts"]))

    def test_promote_refuses_to_write_an_empty_profile(self):
        draft = verify_profile.auto_verify(
            {"facts": [dict(TRUE_FACT, source_span="paraphrased beyond recognition entirely")]},
            RESUME_TEXT)
        with self.assertRaises(ValueError):
            verify_profile.promote(draft, self.tmp, allow_unverified=True)


class DraftConstructionIsDefensive(unittest.TestCase):
    def test_model_cannot_promote_its_own_output(self):
        """A model that helpfully returns verified:true must be overridden."""
        extracted = {"ok": True, "facts": [dict(TRUE_FACT, verified=True)]}
        draft = import_resume.build_draft(extracted, RESUME_TEXT)
        self.assertFalse(draft["facts"][0]["verified"])

    def test_duplicate_fact_ids_do_not_lose_a_fact(self):
        extracted = {"ok": True, "facts": [dict(TRUE_FACT), dict(TRUE_FACT)]}
        draft = import_resume.build_draft(extracted, RESUME_TEXT)
        self.assertEqual(len(draft["facts"]), 2)
        self.assertEqual(len({f["fact_id"] for f in draft["facts"]}), 2)

    def test_phrasings_default_to_the_resumes_own_wording(self):
        extracted = {"ok": True, "facts": [{k: v for k, v in TRUE_FACT.items()}]}
        draft = import_resume.build_draft(extracted, RESUME_TEXT)
        self.assertEqual(draft["facts"][0]["phrasings"], [TRUE_FACT["source_span"]])

    def test_constraints_are_seeded_so_the_denylist_is_not_implicit(self):
        draft = import_resume.build_draft({"ok": True, "facts": []}, RESUME_TEXT)
        self.assertIn("never_claim", draft["constraints"])

    def test_write_draft_refuses_to_touch_the_verified_file(self):
        with self.assertRaises(ValueError):
            import_resume.write_draft({"facts": []}, import_resume.VERIFIED_PATH)

    def test_extraction_failure_is_reported_not_raised(self):
        result = import_resume.extract_facts(
            RESUME_TEXT, chat=stub_chat(None, ok=False, error="429 rate limited"))
        self.assertFalse(result["ok"])
        self.assertIn("429", result["error"])


class EndToEndOnAStubbedModel(unittest.TestCase):
    def test_import_then_verify_produces_a_promotable_profile(self):
        tmp = Path(tempfile.mkdtemp())
        pdf_stub = tmp / "resume.pdf"
        pdf_stub.write_bytes(b"%PDF-1.4 stub")

        extracted = import_resume.extract_facts(
            RESUME_TEXT, chat=stub_chat({"facts": [dict(TRUE_FACT)]}))
        draft = import_resume.build_draft(extracted, RESUME_TEXT, pdf_stub)
        written = import_resume.write_draft(draft, tmp / "draft.json")
        self.assertTrue(written.exists())

        reloaded = json.loads(written.read_text(encoding="utf-8"))
        verified = verify_profile.auto_verify(reloaded, RESUME_TEXT)
        path = verify_profile.promote(verified, tmp / "master_profile.json")

        final = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(final["_status"], "VERIFIED")
        self.assertEqual(len(final["facts"]), 1)

    def test_a_scan_with_no_text_layer_fails_loudly(self):
        tmp = Path(tempfile.mkdtemp())
        pdf_stub = tmp / "resume.pdf"
        pdf_stub.write_bytes(b"%PDF-1.4 stub")

        original = import_resume.read_pdf_text
        import_resume.read_pdf_text = lambda path: "   "
        try:
            result = import_resume.import_resume(pdf_stub, chat=stub_chat({}))
        finally:
            import_resume.read_pdf_text = original
        self.assertFalse(result["ok"])
        self.assertIn("scan", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
