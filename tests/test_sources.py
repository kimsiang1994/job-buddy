"""Offline tests for the multi-source layer.

    py -m unittest tests.test_sources

No network. Every fixture is trimmed from a real response captured on
2026-07-19, because the shapes differ per vendor in ways that are easy to get
wrong from documentation alone -- SmartRecruiters publishes no apply URL at
all, Workable nests location three levels down, Workday packs three values into
one token.
"""

from __future__ import annotations

import unittest

from jobbuddy import job_schema, scoring, source_aggregator, source_ats, source_hn
from jobbuddy import source_workable as workable

# Trimmed from jobs.workable.com/api/v1/jobs?location=Singapore
WORKABLE_RECORD = {
    "id": "727b1cbe-1560-40fe-babf-090ade4f636e",
    "title": "Principal Software Engineer, AI & Data Platform",
    "description": "<p>Build <strong>AI infrastructure</strong>.</p>",
    "requirementsSection": "Python, Kubernetes, 8+ years",
    "company": {"title": "Xora Innovation", "website": "https://xora.vc/"},
    "location": {"city": "Singapore", "subregion": "Singapore",
                 "countryName": "Singapore"},
    "employmentType": "Full-time", "workplace": "hybrid",
    "created": "2026-07-16T11:52:42.978Z", "state": "published",
    "department": "XVL",
    "url": "https://jobs.workable.com/view/abc/principal-software-engineer",
}

# Trimmed from api.smartrecruiters.com -- note there is NO url/absolute_url key.
SMARTRECRUITERS_RECORD = {
    "id": "744000138250539",
    "name": "Senior Data Engineer",
    "ref": "https://api.smartrecruiters.com/v1/companies/BoschGroup/postings/744000138250539",
    "releasedDate": "2026-07-17T05:40:35.752Z",
    "location": {"city": "Singapore", "country": "sg", "remote": False},
    "company": {"identifier": "BoschGroup", "name": "Bosch Group"},
}

GREENHOUSE_RECORD = {
    "id": 4567890,
    "title": "Machine Learning Engineer",
    "absolute_url": "https://boards.greenhouse.io/stripe/jobs/4567890",
    "content": "&lt;p&gt;Build models.&lt;/p&gt;",
    "location": {"name": "Singapore"},
    "first_published": "2026-06-01T00:00:00Z",
}


class LocationClassification(unittest.TestCase):
    """The bug: 'US-Remote' was classified as a Singapore job.

    The first version read `is_overseas = ... and "remote" not in lowered`, so
    any overseas posting mentioning remote work passed the Singapore filter.
    'Remote' says nothing about country until a region is named beside it.
    """

    def test_names_singapore(self):
        self.assertFalse(source_ats.classify_location("Singapore")[0])
        self.assertFalse(source_ats.classify_location("Singapore, Singapore")[0])

    def test_overseas_city_is_overseas_even_with_remote(self):
        overseas, basis = source_ats.classify_location("SF, NYC, remote")
        self.assertTrue(overseas, f"basis={basis}")

    def test_region_scoped_remote_is_overseas(self):
        for text in ("US-Remote", "Remote - US", "UK Remote", "EMEA"):
            with self.subTest(location=text):
                self.assertTrue(source_ats.classify_location(text)[0], text)

    def test_bare_remote_is_kept_not_dropped(self):
        # Could be APAC-eligible. Keeping it lets the reader judge; guessing
        # 'overseas' silently deletes viable remote roles.
        overseas, basis = source_ats.classify_location("Remote")
        self.assertFalse(overseas)
        self.assertIn("region unstated", basis)

    def test_unknown_location_is_kept(self):
        self.assertFalse(source_ats.classify_location("")[0])
        self.assertFalse(source_ats.classify_location("Planet Zog")[0])

    def test_named_overseas_countries(self):
        for text in ("Bengaluru, India", "London", "Tokyo, Japan", "Sydney"):
            with self.subTest(location=text):
                self.assertTrue(source_ats.classify_location(text)[0], text)


class AtsUrlConstruction(unittest.TestCase):
    """Several vendors publish no apply URL, only an API ref."""

    def test_smartrecruiters_gets_a_public_url(self):
        """Every SmartRecruiters job was silently dropped by validate_job.

        Its `ref` field is an API endpoint returning JSON, not a page a human
        can apply on, so the adapter left url empty and validation rejected the
        lot -- 14 Singapore jobs reported as 0.
        """
        job = source_ats.to_job(SMARTRECRUITERS_RECORD, "smartrecruiters",
                                "Bosch Group", "BoschGroup")
        self.assertIsNotNone(job)
        self.assertTrue(job["url"].startswith("https://jobs.smartrecruiters.com/"))
        self.assertNotIn("api.", job["url"])
        self.assertEqual(job_schema.validate_job(job), [])

    def test_greenhouse_keeps_its_own_url(self):
        job = source_ats.to_job(GREENHOUSE_RECORD, "greenhouse", "Stripe", "stripe")
        self.assertEqual(job["url"], GREENHOUSE_RECORD["absolute_url"])

    def test_workday_token_unpacks_into_three_parts(self):
        record = {"title": "Data Engineer", "bulletFields": ["R-123"],
                  "externalPath": "/job/Singapore/Data-Engineer_R-123",
                  "locationsText": "Singapore"}
        job = source_ats.to_job(record, "workday", "OCBC", "ocbc|wd102|External")
        self.assertIsNotNone(job)
        self.assertIn("ocbc.wd102.myworkdayjobs.com", job["url"])


class AtsDetection(unittest.TestCase):
    def test_detects_each_vendor_from_a_url(self):
        cases = [
            ("https://boards.greenhouse.io/stripe", "greenhouse", "stripe"),
            ("https://jobs.lever.co/Coda", "lever", "Coda"),
            ("https://jobs.ashbyhq.com/airwallex", "ashby", "airwallex"),
            ("https://careers.smartrecruiters.com/BoschGroup", "smartrecruiters", "BoschGroup"),
            ("https://acme.recruitee.com/careers", "recruitee", "acme"),
            ("https://acme.breezy.hr/", "breezy", "acme"),
        ]
        for url, vendor, token in cases:
            with self.subTest(url=url):
                found = source_ats.detect_ats(url)
                self.assertEqual(found, (vendor, token))

    def test_workday_packs_tenant_datacentre_and_site(self):
        found = source_ats.detect_ats("https://ocbc.wd102.myworkdayjobs.com/en-US/External")
        self.assertIsNotNone(found)
        vendor, token = found
        self.assertEqual(vendor, "workday")
        self.assertEqual(len(token.split("|")), 3)

    def test_unrelated_url_detects_nothing(self):
        self.assertIsNone(source_ats.detect_ats("https://example.com/about"))

    def test_lever_token_case_is_preserved(self):
        # Lever tokens are case-sensitive: 'Coda' resolves, 'coda' 404s.
        _, token = source_ats.detect_ats("https://jobs.lever.co/Coda")
        self.assertEqual(token, "Coda")


class WorkableMapping(unittest.TestCase):
    def test_maps_a_real_record(self):
        job = workable.to_job(WORKABLE_RECORD)
        self.assertIsNotNone(job)
        self.assertEqual(job["company"], "Xora Innovation")
        self.assertEqual(job["seniority"], "principal")
        self.assertFalse(job["is_overseas"])
        self.assertEqual(job_schema.validate_job(job), [])

    def test_requirements_section_reaches_the_jd_text(self):
        # The skill terms live there; losing it guts the skill match.
        job = workable.to_job(WORKABLE_RECORD)
        self.assertIn("Kubernetes", job["jd_text"])

    def test_salary_is_absent_not_zero(self):
        """Workable publishes no salary. Absent must not read as 'free'."""
        job = workable.to_job(WORKABLE_RECORD)
        self.assertFalse(job["salary_is_stated"])
        self.assertIsNone(job["salary_min_sgd"])

    def test_company_website_is_exposed_for_discovery(self):
        self.assertEqual(workable.company_website(WORKABLE_RECORD), "https://xora.vc/")


class SparseJobsDoNotOutrankRichOnes(unittest.TestCase):
    """The cross-source ranking bug, and the fix.

    Renormalisation is right within one source and wrong across several. MCF
    publishes salary and a real application count; Workable, the ATS boards and
    HN publish neither -- so a sparse job was scored only on the components
    where it happened to do well, which are freshness and low application
    friction. Measured live: nine no-salary jobs scored 90-95 while a role with
    a stated 10-20k range scored 82.
    """

    CONFIG = {
        "filters": {},
        "profile": {"target_seniority": "senior", "years_experience": 5,
                    "skills": {"expert": ["python", "machine learning"]}},
        "weights": {"skill_match": 30, "seniority_fit": 15, "comp_signal": 15,
                    "competition": 20, "company_signal": 10,
                    "application_friction": 5, "freshness": 5},
    }

    def _job(self, **overrides):
        job = job_schema.new_job("test", overrides.pop("key", "x"))
        job.update({"title": "ML Engineer", "company": "Acme", "url": "https://e.test/j",
                    "jd_text": "Build models.", "seniority": "senior",
                    "seniority_basis": "title", "posted_at": "2026-07-18",
                    "is_open": True, "vacancies": 1})
        job.update(overrides)
        return job_schema.finalise(job)

    def test_sparse_job_gets_lower_confidence(self):
        rich = self._job(key="rich", salary_min_sgd=12000, salary_max_sgd=18000,
                         salary_is_stated=True, applications=5, views=100,
                         skills_raw=["Python", "Machine Learning"])
        sparse = self._job(key="sparse", skills_raw=["Python", "Machine Learning"])
        rich_scores = scoring.score_job(rich, self.CONFIG)
        sparse_scores = scoring.score_job(sparse, self.CONFIG)
        self.assertGreater(rich_scores["confidence"], sparse_scores["confidence"])

    def test_missing_data_cannot_inflate_the_rank(self):
        """A perfect score on half the evidence must not beat a good score on all."""
        sparse = self._job(key="sparse", skills_raw=["Python", "Machine Learning"])
        rich = self._job(key="rich", salary_min_sgd=12000, salary_max_sgd=18000,
                         salary_is_stated=True, applications=4, views=90,
                         skills_raw=["Python", "Machine Learning"])
        sparse_scores = scoring.score_job(sparse, self.CONFIG)
        rich_scores = scoring.score_job(rich, self.CONFIG)

        # The sparse job may well score higher on what could be measured...
        if sparse_scores["total"] > rich_scores["total"]:
            # ...but must not outrank the rich one on that basis alone.
            self.assertLessEqual(
                sparse_scores["adjusted"], rich_scores["adjusted"] + 1.0,
                "a job with less evidence outranked one with more",
            )

    def test_adjusted_shrinks_toward_the_neutral_prior(self):
        sparse = self._job(key="s", skills_raw=["Python"])
        scores = scoring.score_job(sparse, self.CONFIG)
        total, adjusted = scores["total"], scores["adjusted"]
        self.assertLess(abs(adjusted - scoring.NEUTRAL_PRIOR), abs(total - scoring.NEUTRAL_PRIOR) + 0.01)

    def test_full_confidence_leaves_the_score_alone(self):
        rich = self._job(key="r", salary_min_sgd=12000, salary_max_sgd=18000,
                         salary_is_stated=True, applications=4, views=90,
                         skills_raw=["Python", "Machine Learning"])
        scores = scoring.score_job(rich, self.CONFIG, velocity={
            "acme": {"open_reqs": 3, "sufficient": True, "history_days": 40,
                     "new_in_window": 1}})
        self.assertAlmostEqual(scores["confidence"], 1.0, places=2)
        self.assertAlmostEqual(scores["total"], scores["adjusted"], places=1)


class HnParsing(unittest.TestCase):
    """HN comments are prose. Parse strictly or skip -- never guess."""

    def test_parses_the_pipe_convention(self):
        parsed = source_hn.parse_posting(
            "Acme Corp | Senior ML Engineer | Singapore or Remote | $180-250k")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["company"], "Acme Corp")
        self.assertEqual(parsed["title"], "Senior ML Engineer")

    def test_prose_without_the_convention_is_skipped(self):
        """A wrong employer on a job list is worse than a missing row."""
        for text in ("We are hiring engineers, email me at bob@acme.com",
                     "Anyone know if this company is any good?",
                     "Acme Corp is great"):
            with self.subTest(text=text[:30]):
                self.assertIsNone(source_hn.parse_posting(text))

    def test_html_entities_are_decoded(self):
        parsed = source_hn.parse_posting(
            "Acme &amp; Co | ML Engineer | Singapore")
        self.assertIn("&", parsed["company"])


class AggregatorGuards(unittest.TestCase):
    def test_reports_unavailable_without_keys(self):
        import os

        saved = {k: os.environ.pop(k, None) for k in
                 ("JSEARCH_API_KEY", "ADZUNA_APP_ID", "ADZUNA_APP_KEY")}
        try:
            jobs, counters = source_aggregator.fetch_jobs("ml engineer")
            self.assertEqual(jobs, [])
            self.assertEqual(counters.get("skipped_no_key"), 1)
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_adzuna_attribution_travels_with_the_data(self):
        """Adzuna's terms make the badge mandatory wherever results are shown.

        Returning it from the data layer means the obligation cannot be
        forgotten when the output format changes.
        """
        notices = source_aggregator.attribution_notices([{"source": "adzuna"}])
        self.assertTrue(any("Adzuna" in n for n in notices))
        self.assertEqual(source_aggregator.attribution_notices([{"source": "mcf"}]), [])


class FetcherTiers(unittest.TestCase):
    """Which strategy a host gets, and which are refused outright."""

    def test_keyless_hosts_use_plain_http(self):
        from jobbuddy import fetcher

        self.assertEqual(fetcher.strategy_for("https://api.mycareersfuture.gov.sg/v2/jobs"), "http")

    def test_measured_browser_hosts_use_a_browser(self):
        from jobbuddy import fetcher

        self.assertEqual(fetcher.strategy_for("https://www.techinasia.com/jobs"), "browser")

    def test_challenge_walled_hosts_need_an_unblocker(self):
        """Sites that block browsers but do NOT name this crawler.

        Indeed used to be in this list and has moved to `never`: it names
        ClaudeBot and anthropic-ai in robots.txt, which is a different thing
        from a generic bot wall and is not something to buy a way around.
        """
        from jobbuddy import fetcher

        for url in ("https://glints.com/sg/jobs", "https://www.glassdoor.sg/Job",
                    "https://www.fastjobs.sg/jobs"):
            with self.subTest(url=url):
                self.assertEqual(fetcher.strategy_for(url), "unblocker")

    def test_linkedin_is_never_attempted(self):
        """Not a technical limit. The account at risk is the job seeker's own."""
        from jobbuddy import fetcher

        self.assertEqual(fetcher.strategy_for("https://www.linkedin.com/jobs"), "never")
        result = fetcher.fetch_page("https://www.linkedin.com/jobs/view/123")
        self.assertFalse(result.ok)
        self.assertEqual(result.strategy, "never")

    def test_challenge_page_is_detected(self):
        from jobbuddy import fetcher

        self.assertTrue(fetcher._looks_challenged(
            "<html><title>Just a moment...</title><body>cf_chl</body></html>"))
        self.assertFalse(fetcher._looks_challenged(
            "<html><body>Senior ML Engineer, Singapore</body></html>"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class SkillExtraction(unittest.TestCase):
    """Reading skills out of prose, for the sources that publish none.

    Before this existed, 71% of jobs on a live run scored nothing on skill
    match -- the heaviest component at weight 30 -- while carrying thousands of
    characters of description nobody read.
    """

    JD = ("Build production ML systems.\n"
          "REQUIREMENTS\n"
          "Python\n"
          "AWS\n"
          "NICE TO HAVE\n"
          "Kubernetes\n"
          "BENEFITS\n"
          "Free lunch and a LangChain t-shirt\n")

    def setUp(self):
        from jobbuddy import skill_extract

        self.sx = skill_extract
        self.vocab = {"Python": 5, "AWS": 5, "Kubernetes": 5, "LangChain": 5}

    def _extract(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            self.sx.save_vocab(self.vocab, path)
            self.sx.reload_vocab()
            try:
                return self.sx.extract(self.JD, path=path)
            finally:
                self.sx.reload_vocab()

    def test_required_and_optional_are_distinguished(self):
        """An optional Kubernetes must not weigh as much as a mandatory Python."""
        found = {s.term: s.required for s in self._extract()}
        self.assertTrue(found.get("Python"))
        self.assertTrue(found.get("AWS"))
        self.assertFalse(found.get("Kubernetes", True))

    def test_skills_under_benefits_are_ignored(self):
        """'LangChain t-shirt' is not a requirement."""
        self.assertNotIn("LangChain", {s.term for s in self._extract()})

    def test_structured_skills_are_never_overwritten(self):
        """MCF's own tags beat anything read out of prose."""
        job = {"skills_raw": ["Machine Learning"], "jd_text": self.JD}
        self.sx.enrich(job)
        self.assertEqual(job["skills_raw"], ["Machine Learning"])

    def test_document_frequency_uses_word_boundaries(self):
        """Substring counting scored POS at 69% -- it matches inside 'position'.

        Every short term was inflated the same way, and the filter then removed
        real skills on the strength of numbers that measured nothing.
        """
        pattern = self.sx._boundary_pattern("POS")
        self.assertIsNone(pattern.search("this position requires"))
        self.assertIsNotNone(pattern.search("experience with POS systems"))

    def test_known_skills_survive_the_frequency_filter(self):
        """Python appeared in 73% of ads and was being discarded as filler.

        The most important skills in a field are the most common ones in that
        field's job ads, so frequency alone is the wrong test.
        """
        record = {"terms": {"Python": 100}, "doc_freq": {"Python": 99},
                  "documents": 100}
        self.assertTrue(self.sx.is_discriminative("Python", record))

    def test_unknown_filler_is_filtered(self):
        record = {"terms": {"Fast-paced": 50}, "doc_freq": {"Fast-paced": 45},
                  "documents": 100}
        self.assertFalse(self.sx.is_discriminative("Fast-paced", record))

    def test_filter_stays_off_on_a_small_corpus(self):
        record = {"terms": {"Whatever": 3}, "doc_freq": {"Whatever": 3},
                  "documents": 4}
        self.assertTrue(self.sx.is_discriminative("Whatever", record))


class HtmlStructurePreserved(unittest.TestCase):
    """Flattening a description to one line destroys its sections.

    `<h3>Nice to have</h3>` is what separates a demand from a wish, and
    collapsing it inline made every extracted skill read as mandatory.
    """

    def test_block_tags_become_line_breaks(self):
        from jobbuddy import html_text

        text = html_text.flatten_html("<p>One</p><h3>Two</h3><ul><li>Three</li></ul>")
        self.assertEqual([line for line in text.split("\n") if line],
                         ["One", "Two", "Three"])

    def test_single_line_mode_still_available(self):
        from jobbuddy import html_text

        text = html_text.flatten_html("<p>One</p><p>Two</p>", preserve_blocks=False)
        self.assertNotIn("\n", text)

    def test_norm_jd_text_keeps_lines_but_collapses_spaces(self):
        cleaned = job_schema.norm_jd_text("A   B\n\n\nC  \n  D")
        self.assertEqual(cleaned, "A B\nC\nD")

    def test_norm_text_still_collapses_everything(self):
        # Titles must stay on one line; only descriptions keep structure.
        self.assertEqual(job_schema.norm_text("A\nB   C"), "A B C")


class RobotsNamedBlocks(unittest.TestCase):
    """A site naming this crawler in robots.txt has answered a question.

    The first parser assumed one User-agent per rule group. robots.txt permits
    STACKING them, and Indeed does exactly that -- a dozen agents listed before
    a single Disallow. The parser captured nothing for a stacked group and
    reported Indeed's job paths as ALLOWED, when Indeed had refused this
    crawler by name.

    A safety check that fails open is worse than no check, because it is the
    one you stop verifying.
    """

    def setUp(self):
        from jobbuddy import site_recon

        self.recon = site_recon

    def test_stacked_user_agents_are_all_detected(self):
        robots = (
            "User-agent: *\n"
            "Disallow: /search\n"
            "\n"
            "User-agent: GPTBot\n"
            "User-agent: CCBot\n"
            "User-agent: anthropic-ai\n"
            "User-agent: ClaudeBot\n"
            "Disallow: /jobs\n"
        )
        found = self.recon.named_blocks(robots)
        self.assertIn("ClaudeBot", found)
        self.assertIn("anthropic-ai", found)

    def test_single_agent_group_still_detected(self):
        self.assertEqual(self.recon.named_blocks("User-agent: ClaudeBot\nDisallow: /\n"),
                         ["ClaudeBot"])

    def test_named_but_allowed_is_not_a_block(self):
        """Being mentioned is not being refused."""
        robots = ("User-agent: ClaudeBot\nAllow: /\n\n"
                  "User-agent: *\nDisallow: /admin\n")
        self.assertEqual(self.recon.named_blocks(robots), [])

    def test_generic_rules_are_not_a_named_block(self):
        self.assertEqual(self.recon.named_blocks("User-agent: *\nDisallow: /admin\n"), [])

    def test_comments_do_not_confuse_the_parser(self):
        robots = ("# our policy\n"
                  "User-agent: ClaudeBot   # no AI crawlers\n"
                  "Disallow: /jobs\n")
        self.assertIn("ClaudeBot", self.recon.named_blocks(robots))

    def test_recon_refuses_to_proceed_past_a_named_block(self):
        report = self.recon.Recon(url="https://example.test",
                                  named_blocks=["ClaudeBot"])
        strategy, why = report.recommendation()
        self.assertEqual(strategy, "do not scrape")
        self.assertIn("explicit refusal", why)


class NamedBlockHostsAreExcluded(unittest.TestCase):
    """Sites that named this crawler are excluded before the unblocker tier.

    A generic bot wall is a site defending itself against load. Naming an agent
    is a site answering a question. Buying an unblocker to get past the second
    one is overriding an explicit answer.
    """

    def test_named_block_hosts_are_never_fetched(self):
        from jobbuddy import fetcher

        for url in ("https://sg.indeed.com/jobs?q=ml",
                    "https://nodeflair.com/jobs",
                    "https://sg.jobstreet.com/jobs",
                    "https://sg.jora.com/jobs"):
            with self.subTest(url=url):
                self.assertEqual(fetcher.strategy_for(url), "never")

    def test_a_named_block_host_cannot_reach_the_unblocker(self):
        import os

        from jobbuddy import fetcher

        os.environ["SCRAPING_PROVIDER"] = "scrapingbee"
        os.environ["SCRAPING_API_KEY"] = "test-key-not-real"
        try:
            result = fetcher.fetch_page("https://sg.indeed.com/jobs")
            self.assertFalse(result.ok)
            self.assertEqual(result.strategy, "never")
        finally:
            os.environ.pop("SCRAPING_PROVIDER", None)
            os.environ.pop("SCRAPING_API_KEY", None)

    def test_cloudflare_only_hosts_still_reach_the_unblocker(self):
        """Glints blocks browsers but does NOT name this crawler.

        robots.txt permits its job detail pages. That is a different situation
        from a named refusal, and must stay reachable via a paid vendor.
        """
        from jobbuddy import fetcher

        self.assertEqual(fetcher.strategy_for("https://glints.com/sg/jobs"),
                         "unblocker")
