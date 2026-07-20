"""Tests for the deterministic career analysis.

    py -m unittest tests.test_career_paths

Offline, fixtures only, no network and no API key. Every company, title, skill
and salary below is invented -- this repo is public and must never carry real CV
or pay data.

The assertions come in two kinds, and the second kind is the reason the module
exists.

**Behaviour.** Clusters form, medians compute, an empty corpus returns an empty
result rather than a traceback.

**Honesty.** These are the ones worth mutation-checking, because each guards a
way the analysis could produce a confident answer it has not earned:

  - a median never escapes a sample smaller than the configured minimum;
  - a thin cluster is visibly thin, so a renderer cannot draw it like a thick one;
  - too little history reports its own insufficiency instead of a trend;
  - the skills gap never names a skill the profile actually holds;
  - the causation and coverage caveats are IN the returned structure, so a later
    renderer edit cannot quietly drop them.

Clustering determinism is asserted by running the same input twice and
comparing, not by eyeballing one output -- the failure this guards against is a
set or dict iteration leaking into the result, which a single run cannot show.
"""

from __future__ import annotations

import unittest

from jobbuddy import career_paths


# --------------------------------------------------------------------------
# fixtures -- all invented
# --------------------------------------------------------------------------

PROFILE = {
    "skills": {
        "expert": ["python", "sql", "machine learning"],
        "working": ["docker", "airflow"],
        "familiar": ["pytorch"],
    },
    "credentials_held": ["bachelor's degree"],
}

CONFIG = {"profile": {"current_salary_sgd_monthly": 9000}}


def job(title: str, *, skills: list[str] | None = None,
        low: int | None = None, high: int | None = None,
        stated: bool = True, seniority: str = "senior",
        key: str = "") -> dict:
    """One posting. Fictional employer, fictional pay."""
    return {
        "job_key": key or f"fake:{title}:{low}",
        "title": title,
        "title_norm": title.lower(),
        "company": "Nimbus Fictional Pte Ltd",
        "seniority": seniority,
        "salary_min_sgd": low,
        "salary_max_sgd": high,
        "salary_is_stated": stated and low is not None,
        "skills_raw": skills if skills is not None else ["Python", "SQL"],
        # Flagged as key, which is what makes them MANDATORY requirements
        # rather than terms the ad merely lists. Without this every cluster's
        # coverage is `None` and correctly renders "not measured" -- `is_core`
        # counts positive evidence only, because counting every extracted term
        # made `core_score` mean "coverage of everything the ad mentions"
        # while its name, and `score_reach`, said mandatory.
        "skills_key": skills if skills is not None else ["Python", "SQL"],
        "jd_text": "You will build things. Python is required.",
        "scope": "test-scope",
    }


def thick_cluster(n: int, title: str = "Data Engineer", **kwargs) -> list[dict]:
    return [job(title, key=f"fake:{title}:{i}", **kwargs) for i in range(n)]


def sighting(run_id: str, ts: str, title: str, key: str) -> dict:
    return {"run_id": run_id, "ts": ts, "title_norm": title, "job_key": key}


# --------------------------------------------------------------------------

class ClusteringIsDeterministic(unittest.TestCase):
    """Same corpus in, byte-identical clusters out -- every time.

    A dict or set iteration leaking into the ordering would show up here and
    nowhere else: one run of a non-deterministic clusterer looks perfectly fine.
    """

    def corpus(self) -> list[dict]:
        return (thick_cluster(6, "Machine Learning Engineer")
                + thick_cluster(5, "Senior Data Scientist")
                + thick_cluster(4, "AI Engineer")
                + thick_cluster(3, "Data Engineer")
                + [job("Chief Bottle Washer")])

    def test_two_runs_agree(self):
        first = career_paths.analyse(self.corpus(), PROFILE, config=CONFIG)
        second = career_paths.analyse(self.corpus(), PROFILE, config=CONFIG)
        self.assertEqual(
            [(c["label"], c["n"], c["rank_score"]) for c in first["clusters"]],
            [(c["label"], c["n"], c["rank_score"]) for c in second["clusters"]],
        )
        self.assertEqual(first, second)

    def test_input_order_does_not_change_the_labels(self):
        """Reordering the postings must not rename or resize a cluster."""
        forward = career_paths.analyse(self.corpus(), PROFILE, config=CONFIG)
        backward = career_paths.analyse(list(reversed(self.corpus())), PROFILE,
                                        config=CONFIG)
        self.assertEqual({(c["label"], c["n"]) for c in forward["clusters"]},
                         {(c["label"], c["n"]) for c in backward["clusters"]})

    def test_abbreviations_join_one_cluster(self):
        """'AI Engineer' and 'Artificial Intelligence Engineer' are one role.

        Guards the contraction: an earlier version EXPANDED the abbreviation
        instead, and the 3-gram window then sliced across it, inventing a
        cluster called 'intelligence product manager'.
        """
        corpus = (thick_cluster(3, "AI Engineer")
                  + thick_cluster(3, "Artificial Intelligence Engineer"))
        result = career_paths.analyse(corpus, PROFILE, config=CONFIG)
        labels = [c["label"] for c in result["clusters"]]
        self.assertEqual(labels, ["ai engineer"])
        self.assertEqual(result["clusters"][0]["n"], 6)

    def test_a_title_naming_no_role_is_left_out(self):
        """Better uncounted than filed under a role that does not describe it."""
        result = career_paths.analyse([job("Chief Bottle Washer")] * 3,
                                      PROFILE, config=CONFIG)
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["unclustered_postings"], 3)


class ThinClustersAreVisiblyThin(unittest.TestCase):
    """A four-posting cluster must not be able to look like a forty-posting one."""

    def test_thin_flag_and_n_are_carried(self):
        result = career_paths.analyse(thick_cluster(4, "Data Engineer"),
                                      PROFILE, config=CONFIG)
        cluster = result["clusters"][0]
        self.assertEqual(cluster["n"], 4)
        self.assertTrue(cluster["thin"])

    def test_a_thick_cluster_is_not_flagged(self):
        result = career_paths.analyse(thick_cluster(20, "Data Engineer"),
                                      PROFILE, config=CONFIG)
        self.assertFalse(result["clusters"][0]["thin"])


class MediansNeverEscapeATinySample(unittest.TestCase):
    """The single most misleading number this module could emit."""

    def test_three_stated_salaries_produce_no_median(self):
        corpus = [job("Data Engineer", low=10000, high=12000, key=f"k{i}")
                  for i in range(3)]
        pay = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["pay"]
        self.assertIsNone(pay["median_sgd"])
        self.assertIn("not enough stated salaries", pay["reason"])
        self.assertIn("n=3", pay["reason"])
        self.assertIsNone(pay["delta_vs_current_sgd"])

    def test_enough_stated_salaries_produce_a_median_and_a_delta(self):
        corpus = [job("Data Engineer", low=10000, high=12000, key=f"k{i}")
                  for i in range(6)]
        pay = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["pay"]
        self.assertEqual(pay["median_sgd"], 11000)
        self.assertEqual(pay["delta_vs_current_sgd"], 2000)
        self.assertIsNone(pay["reason"])

    def test_the_denominator_is_always_reported(self):
        """A median over 5 of 30 must not read like one over 30 of 30."""
        corpus = ([job("Data Engineer", low=10000, high=12000, key=f"s{i}")
                   for i in range(5)]
                  + [job("Data Engineer", low=None, high=None, stated=False,
                         key=f"u{i}") for i in range(25)])
        pay = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["pay"]
        self.assertEqual(pay["stated_n"], 5)
        self.assertEqual(pay["of_n"], 30)
        self.assertAlmostEqual(pay["stated_fraction"], 5 / 30, places=3)

    def test_a_board_that_states_no_salary_is_not_averaged(self):
        """`salary_is_stated` false means the numbers are a guess, not a range."""
        corpus = [job("Data Engineer", low=10000, high=12000, stated=False,
                      key=f"k{i}") for i in range(10)]
        pay = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["pay"]
        self.assertIsNone(pay["median_sgd"])
        self.assertEqual(pay["stated_n"], 0)

    def test_the_minimum_is_configurable(self):
        corpus = [job("Data Engineer", low=10000, high=12000, key=f"k{i}")
                  for i in range(3)]
        config = dict(CONFIG, career_paths={"min_stated_salaries": 3})
        pay = career_paths.analyse(corpus, PROFILE,
                                   config=config)["clusters"][0]["pay"]
        self.assertEqual(pay["median_sgd"], 11000)


class HistoryIsNeverImputed(unittest.TestCase):
    """Too little history reports insufficiency. It never guesses a direction."""

    def test_no_history_at_all(self):
        result = career_paths.analyse(thick_cluster(10), PROFILE, config=CONFIG)
        movement = result["movement"]
        self.assertIn("insufficient history", movement["status"])
        self.assertNotIn("clusters", movement)

    def test_too_few_runs(self):
        sightings = [sighting(f"run{i}", f"2026-01-{i + 1:02d}T00:00:00Z",
                              "data engineer", "k1") for i in range(4)]
        result = career_paths.analyse(thick_cluster(10), PROFILE,
                                      history=sightings, config=CONFIG)
        self.assertEqual(result["movement"]["status"],
                         "insufficient history (4 runs)")
        self.assertNotIn("clusters", result["movement"])

    def test_enough_runs_but_packed_into_two_days_is_still_insufficient(self):
        """Twenty runs in an afternoon measure polling, not hiring.

        This is the check that separates 'we looked a lot' from 'the market
        moved', and it is the one the live corpus actually trips.
        """
        sightings = [
            sighting(f"run{i}", f"2026-01-01T{i:02d}:00:00Z",
                     "data engineer", f"k{i}")
            for i in range(20)
        ]
        movement = career_paths.analyse(thick_cluster(10), PROFILE,
                                        history=sightings,
                                        config=CONFIG)["movement"]
        self.assertIn("insufficient history", movement["status"])
        self.assertIn("spanning", movement["status"])
        self.assertNotIn("clusters", movement)

    def test_enough_runs_over_enough_days_does_measure(self):
        sightings = []
        for run in range(10):
            # Two distinct vacancies in the later runs, one in the earlier ones.
            keys = ["k1", "k2"] if run >= 5 else ["k1"]
            for key in keys:
                sightings.append(sighting(f"run{run:02d}",
                                          f"2026-01-{run * 4 + 1:02d}T00:00:00Z",
                                          "data engineer", key))
        movement = career_paths.analyse(thick_cluster(10), PROFILE,
                                        history=sightings,
                                        config=CONFIG)["movement"]
        self.assertEqual(movement["status"], "measured")
        entry = next(c for c in movement["clusters"] if c["label"] == "data engineer")
        self.assertEqual(entry["postings_prior"], 1)
        self.assertEqual(entry["postings_recent"], 2)
        self.assertEqual(entry["change"], 1)

    def test_repeated_sightings_of_one_vacancy_are_not_a_trend(self):
        """Distinct job_keys, never rows. Polling harder must change nothing."""
        sightings = []
        for run in range(10):
            repeats = 9 if run >= 5 else 1
            for _ in range(repeats):
                sightings.append(sighting(f"run{run:02d}",
                                          f"2026-01-{run * 4 + 1:02d}T00:00:00Z",
                                          "data engineer", "k1"))
        movement = career_paths.analyse(thick_cluster(10), PROFILE,
                                        history=sightings,
                                        config=CONFIG)["movement"]
        entry = next(c for c in movement["clusters"] if c["label"] == "data engineer")
        self.assertEqual(entry["change"], 0)

    def test_skill_movement_says_it_cannot_be_measured(self):
        """The sightings log has no skills. Saying so beats survivorship bias."""
        sightings = [
            sighting(f"run{run:02d}", f"2026-01-{run * 4 + 1:02d}T00:00:00Z",
                     "data engineer", "k1") for run in range(10)
        ]
        movement = career_paths.analyse(thick_cluster(10), PROFILE,
                                        history=sightings,
                                        config=CONFIG)["movement"]
        self.assertEqual(movement["skills"]["status"], "unavailable")
        self.assertIn("not skills", movement["skills"]["reason"])


class TheSkillsGapNamesOnlyRealGaps(unittest.TestCase):
    """A skill the profile holds must never appear in the development plan."""

    def test_owned_skills_are_absent_from_the_gap(self):
        corpus = thick_cluster(
            10, "Data Engineer",
            skills=["Python", "SQL", "Machine Learning", "Docker", "Airflow",
                    "PyTorch", "Terraform", "Kafka"])
        gap = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["skills_gap"]
        named = {g["canonical"] for g in gap}
        for owned in ("python", "sql", "machine learning", "docker", "airflow",
                      "pytorch"):
            self.assertNotIn(owned, named,
                             f"{owned} is on the profile and must not be a gap")
        self.assertEqual(named, {"terraform", "kafka"})

    def test_a_gap_carries_its_frequency_and_denominator(self):
        corpus = (thick_cluster(6, "Data Engineer", skills=["Python", "Kafka"])
                  + thick_cluster(4, "Data Engineer", skills=["Python"],
                                  low=1, high=1, stated=False))
        gap = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["skills_gap"]
        kafka = next(g for g in gap if g["canonical"] == "kafka")
        self.assertEqual(kafka["postings"], 6)
        self.assertEqual(kafka["of_n"], 10)

    def test_a_one_off_requirement_is_not_a_development_plan(self):
        """'Recurring' means more than one posting, or the list is one advert's mood."""
        corpus = (thick_cluster(9, "Data Engineer", skills=["Python"])
                  + [job("Data Engineer", skills=["Python", "Fortran"], key="odd")])
        gap = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["skills_gap"]
        self.assertNotIn("fortran", {g["canonical"] for g in gap})

    def test_extraction_noise_is_dropped(self):
        """Soft attributes are what made every earlier attempt at this useless."""
        corpus = thick_cluster(
            10, "Data Engineer",
            skills=["Python", "Communication", "Teamwork", "Kafka",
                    "scientific discipline"])
        gap = career_paths.analyse(corpus, PROFILE, config=CONFIG)["clusters"][0]["skills_gap"]
        named = {g["canonical"] for g in gap}
        self.assertEqual(named, {"kafka"})


class CaveatsCannotBeDropped(unittest.TestCase):
    """The warnings live in the DATA, so no renderer edit can lose them.

    A caveat that exists only in a report template is one refactor away from
    disappearing, and the numbers it qualifies would go on being published.
    """

    def test_causation_caveat_is_present_and_says_what_it_must(self):
        result = career_paths.analyse(thick_cluster(10), PROFILE, config=CONFIG)
        causation = result["caveats"]["causation"]
        self.assertIn("NOT", causation)
        self.assertIn("senior", causation.lower())

    def test_coverage_caveat_states_vacancies_not_careers(self):
        result = career_paths.analyse(thick_cluster(10), PROFILE, config=CONFIG)
        coverage = result["caveats"]["coverage"]
        self.assertIn("vacancies, not careers", coverage)
        self.assertIn("did next", coverage)

    def test_every_caveat_survives_an_empty_corpus(self):
        """The empty report is where a reader is most tempted to fill the gap."""
        result = career_paths.analyse([], PROFILE, config=CONFIG)
        for key in ("causation", "coverage", "thin_samples",
                    "salary_denominator", "coverage_metric", "selection"):
            self.assertTrue(result["caveats"].get(key), f"{key} caveat missing")

    def test_no_model_is_claimed_or_used(self):
        result = career_paths.analyse(thick_cluster(10), PROFILE, config=CONFIG)
        self.assertTrue(result["deterministic"])
        self.assertIn("no model", result["method"])


class DegenerateInputsDoNotCrash(unittest.TestCase):

    def test_empty_job_list(self):
        result = career_paths.analyse([], PROFILE, config=CONFIG)
        self.assertEqual(result["n_postings"], 0)
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["n_clusters"], 0)

    def test_none_everywhere(self):
        result = career_paths.analyse(None, None, None, None)
        self.assertEqual(result["clusters"], [])
        self.assertIn("insufficient history", result["movement"]["status"])

    def test_jobs_missing_every_optional_field(self):
        result = career_paths.analyse([{}, {}, {}], PROFILE, config=CONFIG)
        self.assertEqual(result["n_postings"], 3)
        self.assertEqual(result["unclustered_postings"], 3)

    def test_a_profile_with_no_skills_measures_nothing_rather_than_zero(self):
        result = career_paths.analyse(thick_cluster(10), {}, config=CONFIG)
        coverage = result["clusters"][0]["coverage"]
        self.assertIsNone(coverage["median_pct"])
        self.assertEqual(coverage["measured_n"], 0)

    def test_a_verified_master_profile_shape_is_accepted_and_declared(self):
        master = {"skill_groups": [{"label": "Stack",
                                    "items": ["Python", "SQL"]}]}
        result = career_paths.analyse(thick_cluster(10), master, config=CONFIG)
        self.assertIn("skill_groups", result["profile_basis"])
        self.assertIn("working", result["profile_basis"])

    def test_no_current_salary_gives_a_median_without_a_delta(self):
        corpus = [job("Data Engineer", low=10000, high=12000, key=f"k{i}")
                  for i in range(6)]
        pay = career_paths.analyse(corpus, PROFILE, config={})["clusters"][0]["pay"]
        self.assertEqual(pay["median_sgd"], 11000)
        self.assertIsNone(pay["delta_pct"])
        self.assertIn("no current salary", pay["reason"])


class RankingPrefersThicknessOverAFlukyFit(unittest.TestCase):

    def test_a_big_decent_cluster_outranks_a_tiny_perfect_one(self):
        corpus = (thick_cluster(30, "Data Engineer", skills=["Python", "Kafka"])
                  + thick_cluster(2, "Platform Architect", skills=["Python"]))
        result = career_paths.analyse(corpus, PROFILE, config=CONFIG)
        self.assertEqual(result["clusters"][0]["label"], "data engineer")

    def test_an_unmeasurable_cluster_never_leads(self):
        corpus = (thick_cluster(3, "Data Engineer", skills=[])
                  + thick_cluster(3, "Platform Architect", skills=["Python"]))
        result = career_paths.analyse(corpus, PROFILE, config=CONFIG)
        self.assertIsNotNone(result["clusters"][0]["rank_score"])


# --------------------------------------------------------------------------
# the renderer
# --------------------------------------------------------------------------

class TheRendererDegradesAndNeverRaises(unittest.TestCase):

    def setUp(self):
        from jobbuddy import render_career
        self.render_career = render_career
        self.analysis = career_paths.analyse(
            thick_cluster(20, "Data Engineer", low=10000, high=12000)
            + thick_cluster(3, "Platform Architect"),
            PROFILE, config=CONFIG)

    def test_typst_missing_writes_the_typ_source_and_says_so(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.render_career, "_load_typst",
                                   return_value=None):
                result = self.render_career.render(self.analysis, Path(tmp))
            self.assertTrue(result["ok"])
            self.assertEqual(result["degraded"], "typ")
            self.assertEqual(result["path"].suffix, ".typ")
            self.assertIn("typst compile", result["note"])
            self.assertTrue(result["path"].exists())

    def test_charts_are_written_even_without_typst(self):
        import tempfile
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.render_career, "_load_typst",
                                   return_value=None):
                result = self.render_career.render(self.analysis, Path(tmp))
            self.assertTrue(result["charts"])
            for path in result["charts"]:
                ET.parse(path)          # every SVG is well-formed

    def test_the_thin_chart_is_drawn_at_low_confidence(self):
        """Thinness is encoded in the SVG, not in a caption beside it."""
        charts = self.render_career.charts(self.analysis)
        self.assertIn('data-state="low-confidence"', charts["coverage_thin"])
        self.assertNotIn('data-state="low-confidence"', charts["coverage"])

    def test_a_suppressed_median_renders_as_unmeasured_not_as_a_low_bar(self):
        charts = self.render_career.charts(self.analysis)
        self.assertIn('data-state="not-measured"', charts["salary_evidence"])
        self.assertIn("not measured", charts["salary_evidence"])

    def test_the_document_prints_the_caveats(self):
        source = self.render_career.build_typst_source(self.analysis)
        self.assertIn("vacancies, not careers", source)
        self.assertIn("Causation", source)
        self.assertIn("No language model was involved", source)

    def test_an_empty_analysis_still_renders(self):
        empty = career_paths.analyse([], PROFILE, config=CONFIG)
        source = self.render_career.build_typst_source(empty)
        self.assertIn("No clusters", source)
        self.assertIn("vacancies, not careers", source)

    def test_the_document_is_deterministic(self):
        first = self.render_career.build_typst_source(self.analysis)
        second = self.render_career.build_typst_source(self.analysis)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
