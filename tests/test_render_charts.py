"""Tests for the hand-rolled SVG charts.

    py -m unittest tests.test_render_charts

Offline, no network, no API key. Every job, employer and figure below is
invented -- this repo is public and must never carry real CV or salary data.

Two kinds of assertion, and only two. Structure: the output parses as XML and
carries the elements a reader would look for. Honesty: a component with no data
never renders as a bar, and a low-confidence bar never looks like a measured
one. The second kind is the reason this module exists -- `scoring.py` already
fixed the impute-the-missing-value bug at the number level, and a chart is where
it would silently come back.

Nothing here asserts on pixels. Positions are layout, not behaviour.
"""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from jobbuddy import render_charts

SVG_NS = "{http://www.w3.org/2000/svg}"


def components(**overrides) -> dict:
    """A `scores["components"]` block. `None` means the component had no data."""
    base = {
        "skill_match": {"value": 78.0, "weight": 30,
                        "detail": {"matched_count": 7, "total_count": 9}},
        "seniority_fit": {"value": 100.0, "weight": 15, "detail": {}},
        "comp_signal": {"value": None, "weight": 15,
                        "detail": {"reason": "salary not stated"}},
        "competition": {"value": 64.0, "weight": 20, "detail": {}},
    }
    base.update(overrides)
    return base


def scores(confidence: float = 0.85, **overrides) -> dict:
    out = {
        "total": 79.0,
        "adjusted": 74.6,
        "confidence": confidence,
        "components": components(),
    }
    out.update(overrides)
    return out


def parse(svg: str) -> ET.Element:
    """Parse, which is the actual validity assertion -- a malformed SVG raises."""
    return ET.fromstring(svg)


def groups(root: ET.Element) -> list[ET.Element]:
    return list(root.iter(f"{SVG_NS}g"))


def texts(element: ET.Element) -> str:
    return " ".join(t.text or "" for t in element.iter(f"{SVG_NS}text"))


def state_of(root: ET.Element, component: str) -> str:
    for group in groups(root):
        if group.get("data-component") == component:
            return group.get("data-state") or ""
    raise AssertionError(f"no group for {component}")


def group_for(root: ET.Element, component: str) -> ET.Element:
    for group in groups(root):
        if group.get("data-component") == component:
            return group
    raise AssertionError(f"no group for {component}")


class ValidSvg(unittest.TestCase):
    """Every chart parses as XML at every data size, including none."""

    def test_component_bars_parse_with_many_single_and_no_components(self):
        for label, data in (
            ("many", components()),
            ("single", {"skill_match": {"value": 50.0, "weight": 30,
                                        "detail": {}}}),
            ("none", {}),
        ):
            with self.subTest(label):
                root = parse(render_charts.component_bars({
                    "confidence": 0.9, "components": data}))
                self.assertEqual(root.tag, f"{SVG_NS}svg")
                self.assertTrue(root.find(f"{SVG_NS}title") is not None)

    def test_distribution_parses_with_zero_one_and_many_peers(self):
        for label, population in (("empty", []), ("single", [70.0]),
                                  ("many", [float(v) for v in range(0, 100, 3)])):
            with self.subTest(label):
                svg = render_charts.score_distribution(64.0, population)
                root = parse(svg)
                peers = [line for line in root.iter(f"{SVG_NS}line")
                         if line.get("data-role") == "peer"]
                self.assertEqual(len(peers), len(population))

    def test_distribution_with_no_peers_says_so_rather_than_drawing_a_scale(self):
        root = parse(render_charts.score_distribution(64.0, []))
        self.assertIn("no other jobs", texts(root))

    def test_gauge_and_timeline_parse(self):
        parse(render_charts.fit_gauge(70.0, 60.0, 0.9))
        parse(render_charts.posting_timeline({"age_days": 12}))

    def test_charts_for_job_returns_one_parseable_svg_per_chart(self):
        job = {"age_days": 9, "scores": scores()}
        charts = render_charts.charts_for_job(job, [70.0, 40.0])
        self.assertEqual(set(charts),
                         {"components", "fit", "timeline", "distribution"})
        for name, svg in charts.items():
            with self.subTest(name):
                parse(svg)


class Honesty(unittest.TestCase):
    """A chart must never imply precision the data lacks."""

    def test_a_component_with_no_data_draws_no_bar_at_all(self):
        root = parse(render_charts.component_bars(scores()))
        group = group_for(root, "comp_signal")
        self.assertEqual(group.get("data-state"), "not-measured")
        fills = {rect.get("fill") for rect in group.iter(f"{SVG_NS}rect")}
        # No solid fill anywhere in the group: nothing for the eye to compare.
        self.assertNotIn(render_charts.STRONG, fills)
        self.assertNotIn("url(#jb-hatch)", fills)
        self.assertIn("url(#jb-empty)", fills)

    def test_a_component_with_no_data_never_renders_at_mid_range(self):
        """The exact bug `scoring.py` fixed at the number level, at the picture
        level: a missing component drawn at 50 is a claim of "average"."""
        root = parse(render_charts.component_bars(scores()))
        group = group_for(root, "comp_signal")
        widths = [float(r.get("width")) for r in group.iter(f"{SVG_NS}rect")]
        # Only full-width track rects. No partial bar of any length exists,
        # so no mid-range -- or any other -- value can be read off it.
        self.assertEqual(len(set(widths)), 1)
        self.assertIn("not measured", texts(group))

    def test_a_missing_component_states_the_reason_it_could_not_be_scored(self):
        root = parse(render_charts.component_bars(scores()))
        self.assertIn("salary not stated", texts(group_for(root, "comp_signal")))

    def test_low_confidence_bars_are_drawn_differently_from_measured_ones(self):
        high = parse(render_charts.component_bars(scores(confidence=0.9)))
        low = parse(render_charts.component_bars(scores(confidence=0.2)))

        self.assertEqual(state_of(high, "skill_match"), "measured")
        self.assertEqual(state_of(low, "skill_match"), "low-confidence")

        high_fills = {r.get("fill") for r
                      in group_for(high, "skill_match").iter(f"{SVG_NS}rect")}
        low_fills = {r.get("fill") for r
                     in group_for(low, "skill_match").iter(f"{SVG_NS}rect")}
        self.assertIn(render_charts.STRONG, high_fills)
        self.assertNotIn(render_charts.STRONG, low_fills)
        self.assertIn("url(#jb-hatch)", low_fills)

    def test_low_confidence_is_stated_in_words_as_well_as_hatching(self):
        low = parse(render_charts.component_bars(scores(confidence=0.2)))
        self.assertIn("low confidence", texts(low))

    def test_a_gauge_with_no_value_says_insufficient_data_not_zero(self):
        root = parse(render_charts.fit_gauge(None, 60.0, None))
        self.assertEqual(groups(root)[0].get("data-state"), "not-measured")
        self.assertIn("insufficient data", texts(root))

    def test_a_gauge_with_no_stated_bar_draws_no_target_tick(self):
        """An invented target would turn "they did not say" into a pass or fail."""
        without = parse(render_charts.fit_gauge(70.0, None, 0.9))
        ticks = [line for line in without.iter(f"{SVG_NS}line")
                 if line.get("data-role") == "target"]
        self.assertEqual(ticks, [])
        self.assertIn("not measured", texts(without))

        with_target = parse(render_charts.fit_gauge(70.0, 60.0, 0.9))
        ticks = [line for line in with_target.iter(f"{SVG_NS}line")
                 if line.get("data-role") == "target"]
        self.assertEqual(len(ticks), 1)

    def test_a_posting_with_no_date_says_so_rather_than_drawing_day_zero(self):
        root = parse(render_charts.posting_timeline({"age_days": None}))
        self.assertEqual(groups(root)[0].get("data-state"), "not-measured")
        self.assertIn("no posting date", texts(root))
        self.assertEqual(
            [line for line in root.iter(f"{SVG_NS}line")
             if line.get("data-role") == "elapsed"], [])

    def test_a_posting_with_no_closing_date_does_not_invent_one(self):
        root = parse(render_charts.posting_timeline({"age_days": 10}))
        self.assertEqual(
            [c for c in root.iter(f"{SVG_NS}circle")
             if c.get("data-role") == "closes"], [])
        self.assertIn("closing date not measured", texts(root))

    def test_distribution_with_no_score_marks_nothing(self):
        root = parse(render_charts.score_distribution(None, [10.0, 20.0]))
        self.assertEqual([line for line in root.iter(f"{SVG_NS}line")
                          if line.get("data-role") == "this-job"], [])
        self.assertIn("not measured", texts(root))


class NeverCrashes(unittest.TestCase):
    """Missing, None and junk values must not take a renderer down.

    A chart that raises loses the whole report, and the report is the artefact
    that says what could not be measured -- so the failure would delete exactly
    the information the reader needs most.
    """

    def test_none_and_empty_inputs_still_produce_svg(self):
        for label, svg in (
            ("no scores", render_charts.component_bars(None)),
            ("empty scores", render_charts.component_bars({})),
            ("no job", render_charts.posting_timeline(None)),
            ("no population", render_charts.score_distribution(None, None)),
            ("everything none", render_charts.fit_gauge(None, None, None)),
        ):
            with self.subTest(label):
                parse(svg)

    def test_non_numeric_values_are_treated_as_unmeasured_not_as_zero(self):
        root = parse(render_charts.component_bars({
            "confidence": "unknown",
            "components": {"skill_match": {"value": "n/a", "weight": 30}}}))
        self.assertEqual(state_of(root, "skill_match"), "not-measured")

    def test_non_numeric_peers_are_dropped_rather_than_coerced(self):
        """A job that failed to score is not a job that scored zero."""
        root = parse(render_charts.score_distribution(50.0, [70.0, None, "x"]))
        peers = [line for line in root.iter(f"{SVG_NS}line")
                 if line.get("data-role") == "peer"]
        self.assertEqual(len(peers), 1)

    def test_out_of_range_values_are_clamped_rather_than_overflowing(self):
        root = parse(render_charts.component_bars({
            "confidence": 1.0,
            "components": {"a": {"value": 480.0, "weight": 5},
                           "b": {"value": -80.0, "weight": 5}}}))
        for name in ("a", "b"):
            widths = [float(r.get("width"))
                      for r in group_for(root, name).iter(f"{SVG_NS}rect")]
            self.assertLessEqual(max(widths), render_charts.WIDTH)
            self.assertGreaterEqual(min(widths), 0.0)

    def test_markup_in_a_label_is_escaped_so_the_svg_still_parses(self):
        root = parse(render_charts.component_bars({
            "confidence": 1.0,
            "components": {"R&D <script>": {"value": 50.0, "weight": 5}}}))
        self.assertIn("R&D <script>", texts(root))


if __name__ == "__main__":
    unittest.main()
