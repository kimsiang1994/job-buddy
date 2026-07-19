"""HTML to plain text, stdlib only.

Lives in the core because two unrelated things need it: job descriptions
arriving as HTML from MyCareersFuture, and DeepSeek's public docs pages. It
started inside the docs scraper, which meant the search pipeline imported the
whole LLM layer to flatten a job description -- a dependency pointing the wrong
way, from the part that runs for free to the part that costs money.

Regexing flattened *text* rather than the DOM survives a theme change; it only
breaks if the page's actual wording changes.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """Collect visible text, skipping the tags whose contents are not prose."""

    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def flatten_html(html: str) -> str:
    """Whitespace-collapsed visible text. Never raises.

    A malformed job description must degrade to whatever text could be
    recovered, not take down the record it belongs to.
    """
    parser = TextExtractor()
    try:
        parser.feed(html or "")
    except Exception:
        # HTMLParser is tolerant, but a pathological document should not be
        # able to lose a whole job.
        return re.sub(r"<[^>]+>", " ", html or "").strip()
    return parser.text()


# Kept so the docs scraper's existing references still resolve.
_TextExtractor = TextExtractor
