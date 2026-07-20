"""Tests for compound advert-phrase filtering in `skills_taxonomy`.

    py -m unittest tests.test_skill_noise

Offline, fixtures only. Every phrase below was read off the real MyCareersFuture
corpus, but no company, salary or CV content appears here -- this repo is public.

**The bug these guard.** The original noise patterns were anchored `^...$`
against single generic words, so compound advert copy walked straight past them.
Measured on 331 real postings, the top "missing skills" for the largest cluster
were 'Design', 'Development', 'Liaising with cross functional teams',
'Scalability' and 'Data Solutions' -- sitting at the same document frequency as
AI Evaluation, CI/CD and Model Training. Half the development plan was junk.

**The asymmetry, which is the point.** Dropping a term removes it from the
scoring denominator, so a WRONG drop makes coverage read higher than it is and
deletes a genuine gap from the plan -- an error nobody can see. A wrong KEEP is
visible clutter. The tests are therefore weighted the same way: the "real skills
survive" and "ambiguous stays" cases are the load-bearing ones, and
`TheBiasIsTowardKeeping` exists specifically so a later edit that tightens the
filter past the line fails rather than passing quietly.
"""

from __future__ import annotations

import unittest
from unittest import mock

from jobbuddy import skills_taxonomy


class RealSkillsSurvive(unittest.TestCase):
    """The failure mode that matters. Every one of these must be kept."""

    REAL = (
        # named technologies
        "Python", "Kubernetes", "PySpark", "LangChain", "ETL", "C++", ".NET",
        "PyTorch", "AWS", "Docker", "FastAPI", "Pinecone", "scikit-learn",
        # named practices and methods
        "CI/CD", "AI Safety", "Bayesian modelling", "A/B testing",
        "Prompt Engineering", "Model Training", "AI Evaluation",
        "Predictive Analytics", "Feature Engineering", "Data Governance",
        # real skills that LOOK like the shapes the filter hunts
        "High Availability",          # canonicalises onto 'scalability'
        "Scalable Systems",           # ditto
        "Software as a Service (SaaS)",   # long, no technical anchor
        "BIRT (Business Intelligence and Reporting Tools)",   # long product name
        "Data Integrity",             # 'integrity' reads soft, is not
        "Design Patterns",            # starts with a bare abstract noun
        "Test Automation", "Data Wrangling", "Model Deployment",
        "Retrieval-Augmented Generation (RAG)",
        "Generative AI Application Development and Deployment in Financial Services",
    )

    def test_every_real_skill_survives(self):
        for skill in self.REAL:
            with self.subTest(skill=skill):
                self.assertFalse(skills_taxonomy.is_noise(skill), skill)

    def test_they_survive_clean_job_skills_too(self):
        """is_noise is the rule; clean_job_skills is what scoring calls.

        Compared against canonical de-duplication rather than raw length:
        'High Availability' and 'Scalable Systems' both canonicalise onto
        'scalability', so one of them is legitimately collapsed away and the
        naive length check would read that as a loss.
        """
        kept = skills_taxonomy.clean_job_skills(self.REAL)
        expected = {skills_taxonomy.canon(term) for term in self.REAL}
        self.assertEqual({skills_taxonomy.canon(term) for term in kept},
                         expected)


class AdvertPhrasesAreDropped(unittest.TestCase):
    """The five measured offenders, plus the families they belong to."""

    def test_the_five_that_broke_the_development_plan(self):
        for phrase in ("Design", "Development",
                       "Liaising with cross functional teams",
                       "Scalability", "Data Solutions"):
            with self.subTest(phrase=phrase):
                self.assertTrue(skills_taxonomy.is_noise(phrase), phrase)

    def test_gerund_and_verb_led_phrases(self):
        for phrase in ("Liaising With Internal Stakeholders",
                       "Liaise With Technical Teams",
                       "Collaborating With Product Managers",
                       "collaborate with external team",
                       "Working alongside Product Teams",
                       "Engaging Stakeholders", "Communicating With Business",
                       "Build Prototypes", "Create Documentation",
                       "Gather Feedback", "Improve Usability",
                       "Interpret Business Needs", "Solve Technical Problems",
                       "Translate Into Technical Requirements",
                       "Managing Ambiguity", "leading technical teams"):
            with self.subTest(phrase=phrase):
                self.assertTrue(skills_taxonomy.is_noise(phrase), phrase)

    def test_soft_quality_phrases(self):
        for phrase in ("Analytical and Problem-Solving Skills", "Soft Skills",
                       "Presentation Skills", "Technical Skillset",
                       "Computer Proficiency", "Technological Proficiency",
                       "Track Record Of Dependability", "Accountability",
                       "Work in A Fast Paced Environment",
                       "Enthusiasm to learn", "Articulate Communicator"):
            with self.subTest(phrase=phrase):
                self.assertTrue(skills_taxonomy.is_noise(phrase), phrase)

    def test_filler_head_nouns(self):
        for phrase in ("Solution Delivery", "Timely Delivery",
                       "Technology Expertise", "Industry Experience",
                       "Extensive Work Experience", "Professional Experience",
                       "Operations Capability", "Business Environment",
                       "Partner Solutions", "Voice Solutions",
                       "computing solutions", "scalable solutions"):
            with self.subTest(phrase=phrase):
                self.assertTrue(skills_taxonomy.is_noise(phrase), phrase)

    def test_of_constructions(self):
        for phrase in ("knowledge of data networking",
                       "knowledge of the latest developments",
                       "Analysis of Data Sources"):
            with self.subTest(phrase=phrase):
                self.assertTrue(skills_taxonomy.is_noise(phrase), phrase)

    def test_bare_abstract_nouns(self):
        for phrase in ("Management", "Operations", "Adoption", "Budget",
                       "Certifications", "Conferences", "Journals",
                       "Internships", "Education", "Tenure", "Diversity"):
            with self.subTest(phrase=phrase):
                self.assertTrue(skills_taxonomy.is_noise(phrase), phrase)


class KnownTaxonomyTermsAreProtected(unittest.TestCase):
    """A previous filter deleted Python (73% doc frequency), AWS and API by
    frequency alone. The fix was protecting known terms; this is the guard
    against reintroducing that, whatever shape the term arrives in.
    """

    def test_every_alias_and_canonical_survives(self):
        exempt = skills_taxonomy._ABSTRACT_TAXONOMY_TERMS
        for term in skills_taxonomy.ALIASES:
            if term in exempt:
                continue
            with self.subTest(term=term):
                self.assertFalse(skills_taxonomy.is_advert_phrase(term), term)

    def test_protection_beats_shape(self):
        """Taxonomy membership must be checked BEFORE the shape rules.

        No alias shipping today is advert-shaped, so the ordering cannot be
        proved from the real vocabulary -- every one of them would also survive
        on `has_technical_anchor` alone, and a mutation removing the protection
        branch went undetected until this test existed. The contract being
        asserted is "a term in the taxonomy is a skill whatever shape it has",
        which only bites when someone later adds an advert-shaped alias. So one
        is injected here.
        """
        advert_shaped = ("liaising with stakeholders", "attention to detail",
                         "design")
        with mock.patch.dict(skills_taxonomy.ALIASES,
                             {term: term for term in advert_shaped}):
            for term in advert_shaped:
                with self.subTest(term=term):
                    self.assertFalse(skills_taxonomy.is_advert_phrase(term),
                                     term)

    def test_the_injected_terms_really_are_advert_shaped(self):
        """Without the patch above, all three must be dropped.

        Otherwise the previous test proves nothing: it would pass against a
        build with no protection at all.
        """
        for term in ("liaising with stakeholders", "attention to detail",
                     "design"):
            with self.subTest(term=term):
                self.assertTrue(skills_taxonomy.is_advert_phrase(term), term)

    def test_the_one_documented_override_is_narrow(self):
        """'Scalability' bare is advert copy; its aliases are real skills.

        The override is deliberately tested on the surface form only. If a
        later edit moves it back onto the canonical form, 'High Availability'
        and 'Distributed Systems' start disappearing and this fails.
        """
        self.assertTrue(skills_taxonomy.is_advert_phrase("Scalability"))
        for alias in ("High Availability", "Distributed Systems",
                      "Scalable Systems"):
            with self.subTest(alias=alias):
                self.assertEqual(skills_taxonomy.canon(alias), "scalability")
                self.assertFalse(skills_taxonomy.is_advert_phrase(alias), alias)


class TheBiasIsTowardKeeping(unittest.TestCase):
    """Ambiguous terms STAY. Asserted explicitly so a later tightening fails.

    Each of these is arguable -- a case could be made that it is advert copy.
    That case is exactly why it must be kept: a wrong drop hides a real gap,
    a wrong keep is visible clutter. If someone later decides these should go,
    this test going red is the intended conversation.
    """

    AMBIGUOUS = (
        "Vision",            # company vision, or computer vision?
        "Writing",           # bare, but writing is a nameable skill
        "Research", "Testing", "Deployment", "Evaluation", "Validation",
        "Debugging", "Monitoring", "Orchestration", "Optimization",
        "Classification", "Segmentation", "Reproducibility", "Traceability",
        "Metrics", "Patterns", "Prototype", "Workflow",
        "Autonomy iManage",  # 'autonomy' reads soft; this is a product
        "Deep Foundations",  # a real civil-engineering skill, not AI copy
        "Automatic Storage and Retrieval System",
    )

    def test_ambiguous_terms_are_kept(self):
        for term in self.AMBIGUOUS:
            with self.subTest(term=term):
                self.assertFalse(skills_taxonomy.is_noise(term), term)


class TheAnchorRuleIsNotGenerous(unittest.TestCase):
    """`has_technical_anchor` requires a WHOLE taxonomy key, not a shared word.

    The generous version -- "any word of the term appears somewhere in the
    taxonomy" -- anchors 'Data Solutions' on 'data' (from 'data engineering')
    and defeats the entire filter. That regression is silent, so it is asserted
    directly rather than only through its consequences.
    """

    def test_whole_keys_anchor(self):
        for term in ("Enterprise Security Solutions",   # 'security'
                     "Advanced Kubernetes Operations",  # 'kubernetes'
                     "Generative AI Model Evaluation"):  # 'generative ai'
            with self.subTest(term=term):
                self.assertTrue(skills_taxonomy.has_technical_anchor(term), term)

    def test_a_shared_word_does_not_anchor(self):
        # 'Business Delivery Environment' shares 'delivery' with nothing in the
        # taxonomy; note 'Model Governance Environment' would legitimately
        # anchor, because 'governance' IS an alias of 'data governance'.
        for term in ("Data Solutions", "Business Delivery Environment",
                     "Learning Culture"):
            with self.subTest(term=term):
                self.assertFalse(skills_taxonomy.has_technical_anchor(term), term)

    def test_an_anchor_rescues_a_long_phrase(self):
        long_real = ("Generative AI Application Development and Deployment "
                     "in Financial Services")
        long_junk = "Automation Management in Product Development Reviews Team"
        self.assertFalse(skills_taxonomy.is_advert_phrase(long_real))
        self.assertTrue(skills_taxonomy.is_advert_phrase(long_junk))


class CleanJobSkillsStillBehaves(unittest.TestCase):

    def test_order_and_deduplication_survive_the_new_rules(self):
        kept = skills_taxonomy.clean_job_skills(
            ["Python", "Design", "Kubernetes", "python", "Data Solutions",
             "CI/CD"])
        self.assertEqual(kept, ["Python", "Kubernetes", "CI/CD"])

    def test_a_job_of_pure_advert_copy_yields_nothing(self):
        self.assertEqual(
            skills_taxonomy.clean_job_skills(
                ["Design", "Development", "Liaising with cross functional teams",
                 "Soft Skills", "Timely Delivery"]),
            [])

    def test_junk_input_never_raises(self):
        for value in (None, "", "   ", 0, [], {}, object()):
            with self.subTest(value=repr(value)):
                skills_taxonomy.is_advert_phrase(value)
                skills_taxonomy.has_technical_anchor(value)
                skills_taxonomy.surface(value)


if __name__ == "__main__":
    unittest.main()
