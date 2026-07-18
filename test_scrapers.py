"""Regression tests for the docs scrapers in update_models.py.

    py test_scrapers.py

Offline and deterministic -- no network, no API key, no cost. Every fixture here
is real text that actually broke the parser once, so please add to it rather than
rewrite it when the docs change shape again.

Why this file exists: DeepSeek's docs are served with different markup in
different regions, so a scraper that passes locally can still fail in CI. Both
failures below were found that way.
"""

import unittest

import update_models as u

IDS = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]
LEGACY = {"deepseek-chat", "deepseek-reasoner"}


class DeprecationAttribution(unittest.TestCase):
    """Only the models actually being retired may be flagged.

    Deprecation is sticky, so a false positive permanently disqualifies a live
    model from tier selection and needs --clear-deprecation to undo.
    """

    def test_list_form_does_not_retire_neighbours(self):
        # The shape that broke CI: a model LIST, no sentences at all. A proximity
        # window sweeps up the two live models sitting next to deepseek-chat.
        text = ("api_key apply for an API key model * deepseek-v4-flash "
                "deepseek-v4-pro deepseek-chat (to be deprecated on 2026/07/24) "
                "deepseek-reasoner (to be deprecated on 2026/07/24) * The model "
                "names deepseek-chat and deepseek-reasoner will be deprecated on "
                "2026/07/24 15:59 UTC.")
        self.assertEqual(set(u.scrape_deprecations(text, IDS)), LEGACY)

    def test_prose_form_chains_across_and(self):
        text = ("The two legacy API model names, deepseek-chat and "
                "deepseek-reasoner , will be discontinued in three months "
                "(2026-07-24).")
        self.assertEqual(set(u.scrape_deprecations(text, IDS)), LEGACY)

    def test_replacement_named_after_is_not_retired(self):
        # deepseek-v4-flash is the replacement here, not the victim.
        text = ("The model names deepseek-chat and deepseek-reasoner will be "
                "deprecated on 2026/07/24 15:59 UTC. For compatibility, they "
                "correspond to the thinking mode of deepseek-v4-flash , "
                "respectively.")
        self.assertEqual(set(u.scrape_deprecations(text, IDS)), LEGACY)

    def test_adjacent_launch_sentence_does_not_leak(self):
        text = ("DeepSeek-V4 is now available as deepseek-v4-pro and "
                "deepseek-v4-flash . The two legacy API model names, "
                "deepseek-chat and deepseek-reasoner , will be discontinued in "
                "three months (2026-07-24).")
        self.assertEqual(set(u.scrape_deprecations(text, IDS)), LEGACY)

    def test_no_date_is_not_a_notice(self):
        text = "The model deepseek-v4-pro may eventually be deprecated."
        self.assertEqual(u.scrape_deprecations(text, IDS), {})


# A trimmed copy of the real pricing page, flattened. Note the intro sentence:
# it contains the words "output tokens", which is why the row labels must be
# anchored on "1M" -- without that the parser matches the prose instead of the
# table and every price column silently fails to populate.
PRICING_TEXT = (
    "Models & Pricing The prices listed below are in units of per 1M tokens. We "
    "will bill based on the total number of input and output tokens by the "
    "model. Model Details MODEL deepseek-v4-flash (1) deepseek-v4-pro BASE URL "
    "(OpenAI Format) https://api.deepseek.com CONTEXT LENGTH 1M MAX OUTPUT "
    "MAXIMUM: 384K PRICING 1M INPUT TOKENS (CACHE HIT) $0.0028 $0.003625 "
    "1M INPUT TOKENS (CACHE MISS) $0.14 $0.435 1M OUTPUT TOKENS $0.28 $0.87 "
    "Concurrency Limit (2) 2500 500"
)


class PricingTable(unittest.TestCase):
    """The table is transposed: models are columns, prices are row labels."""

    def test_parses_both_columns(self):
        rows = u.scrape_pricing(PRICING_TEXT, IDS)
        self.assertEqual(rows["deepseek-v4-flash"],
                         {"input_cache_hit": 0.0028, "input_cache_miss": 0.14,
                          "output": 0.28})
        self.assertEqual(rows["deepseek-v4-pro"],
                         {"input_cache_hit": 0.003625, "input_cache_miss": 0.435,
                          "output": 0.87})

    def test_rejects_row_with_wrong_column_count(self):
        # A third column appearing means the shape changed; guessing the mapping
        # would be worse than parsing nothing.
        broken = PRICING_TEXT.replace("$0.14 $0.435", "$0.14 $0.435 $0.99")
        self.assertNotIn("deepseek-v4-flash", u.scrape_pricing(broken, IDS))

    def test_rejects_implausible_ordering(self):
        # output >= cache_miss >= cache_hit is the invariant that makes the
        # positional mapping safe.
        self.assertFalse(u._pricing_valid(
            {"input_cache_hit": 9.0, "input_cache_miss": 0.14, "output": 0.28}))


class ModelIdParsing(unittest.TestCase):
    def test_legacy_names_are_never_auto_selectable(self):
        # Returning None here is what keeps deepseek-chat out of tier selection.
        import model_config
        self.assertIsNone(model_config.parse_model_id("deepseek-chat"))
        self.assertIsNone(model_config.parse_model_id("deepseek-reasoner"))
        self.assertEqual(model_config.parse_model_id("deepseek-v4-flash"),
                         {"generation": 4, "minor": 0, "family": "flash"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
