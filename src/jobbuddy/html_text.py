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


# Tags that end a line of prose. Collapsing across these destroys the document
# structure -- and in a job description that structure is meaningful, because
# `<h3>Nice to have</h3>` is what separates a demand from a wish. Flattening it
# inline made every extracted skill read as mandatory.
BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header",
    "footer", "blockquote", "pre", "hr", "dt", "dd", "dl",
}


class TextExtractor(HTMLParser):
    """Collect visible text, skipping the tags whose contents are not prose."""

    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self, preserve_blocks: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0
        self._preserve = preserve_blocks

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif self._preserve and tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif self._preserve and tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts) if self._preserve else " ".join(self._parts)
        if not self._preserve:
            return re.sub(r"\s+", " ", joined).strip()
        # Collapse spaces and tabs, but keep line breaks -- and collapse runs of
        # blank lines, which block tags produce in pairs.
        joined = re.sub(r"[ \t ]+", " ", joined)
        joined = re.sub(r"\n\s*\n+", "\n", joined)
        return "\n".join(line.strip() for line in joined.split("\n")).strip()


def flatten_html(html: str, preserve_blocks: bool = True) -> str:
    """Visible text from HTML. Never raises.

    `preserve_blocks` keeps one line per block element, which is what lets a
    reader (or a section detector) tell a heading from body text. Pass False
    for the old single-line behaviour where structure does not matter.

    A malformed description must degrade to whatever text could be recovered,
    not take down the record it belongs to.

    The regex fallback is a real degradation, not an equivalent path: it loses
    the block structure that tells `skill_extract` a heading from body text, so
    every skill under 'Nice to have' starts reading as mandatory. Taking it
    without a word meant a job could be scored on a silently worse extraction
    and nothing distinguished it from the rest of the run.
    """
    # Coerced once, up front. A source handing this a non-string (an int id, a
    # dict from a malformed API response) used to reach `parser.feed()`, raise
    # TypeError, and then raise a SECOND TypeError inside the handler itself
    # where `len(html)` was interpolated into the warning -- so the function
    # documented as never raising raised, out of its own recovery path.
    if not isinstance(html, str):
        html = "" if html is None else str(html)

    parser = TextExtractor(preserve_blocks=preserve_blocks)
    try:
        parser.feed(html)
    except Exception as exc:
        # Deliberately broad: HTMLParser is tolerant, so anything reaching here
        # is a document shape stdlib did not anticipate, and the point is that
        # a pathological description cannot lose a whole job.
        from jobbuddy import net

        net._warn(f"html: {type(exc).__name__} parsing a {len(html)}-char "
                  f"document ({exc}); falling back to regex tag-stripping, "
                  f"which loses block structure and section awareness")
        return re.sub(r"<[^>]+>", " ", html).strip()
    return parser.text()


# Kept so the docs scraper's existing references still resolve.
_TextExtractor = TextExtractor
