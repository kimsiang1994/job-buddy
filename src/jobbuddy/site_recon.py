"""Work out how to read a job site, before writing any adapter for it.

    py -m jobbuddy.site_recon https://www.mycareersfuture.gov.sg

Guessing at a scraper wastes a day and produces something brittle. Nearly every
job site already publishes its listings in a machine-readable form for somebody
-- its own frontend, Google, or a crawler -- and the job is to find which. In
rough order of how pleasant they are to consume:

  1. robots.txt / sitemap.xml   the site telling you what it permits, in
                                writing. Read this FIRST -- it decides whether
                                anything below is appropriate at all.
  2. JSON-LD JobPosting         schema.org markup embedded in the page. Any
                                site ranking in Google for Jobs must publish
                                it, so this is common and it is structured
                                data the site deliberately exposes.
  3. XHR/fetch to a JSON API    what the site's own frontend calls. This is how
                                MyCareersFuture's v2 API was found.
  4. RSS/Atom                   built for machine consumption. Rare in 2026.
  5. HTML parsing               last resort, and the most fragile.

This module reports what it finds and stops. It does not scrape, and it does
not attempt anything robots.txt disallows -- including where a site names this
crawler specifically, which several Singapore boards now do.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Any

from jobbuddy import net

# Agent tokens worth checking a robots.txt for by name. A site that singles one
# of these out has made a clearer statement than any generic rule.
NAMED_AGENTS = ("ClaudeBot", "anthropic-ai", "Claude-Web", "GPTBot",
                "CCBot", "Google-Extended", "Applebot-Extended", "Bytespider")

OUR_AGENT = "job-buddy"

# Analytics, ads and tag managers. They fire on every page and quote the page
# they fired from, so they look like whatever you are searching for.
THIRD_PARTY_HOSTS = (
    "google-analytics", "googletagmanager", "doubleclick", "ads.linkedin",
    "facebook.net", "facebook.com", "hotjar", "segment.io", "segment.com",
    "mixpanel", "amplitude", "sentry.io", "datadoghq", "newrelic",
    "cloudflareinsights", "clarity.ms", "bing.com", "criteo", "adsrvr",
    "googleadservices", "gstatic", "cookielaw", "onetrust", "intercom",
    "fullstory", "heap.io", "optimizely", "quantserve", "scorecardresearch",
)

JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.S | re.I)
FEED_RE = re.compile(
    r'<link[^>]+type=["\']application/(?:rss|atom)\+xml["\'][^>]*href=["\']([^"\']+)',
    re.I)


@dataclass
class Recon:
    """Everything learned about one site, and the recommended way in."""

    url: str
    host: str = ""
    robots_fetched: bool = False
    jobs_path_allowed: bool | None = None
    named_blocks: list[str] = field(default_factory=list)
    sitemaps: list[str] = field(default_factory=list)
    job_sitemaps: list[str] = field(default_factory=list)
    feeds: list[str] = field(default_factory=list)
    jsonld_types: list[str] = field(default_factory=list)
    has_job_posting_markup: bool = False
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    challenge: bool = False
    notes: list[str] = field(default_factory=list)

    def recommendation(self) -> tuple[str, str]:
        """(strategy, why). The cheapest appropriate route in.

        The decisive test is the `User-agent: *` group, because that is the
        rule governing this tool. `named_blocks` is reported for context but
        does NOT decide: those tokens identify other people's crawlers, and a
        rule aimed at GPTBot does not bind a program that is not GPTBot.
        Treating a named block as a refusal excluded three sites whose wildcard
        group plainly allowed their job paths.
        """
        if self.jobs_path_allowed is False:
            return ("do not scrape",
                    "robots.txt `User-agent: *` disallows the job listing paths -- "
                    "the site is refusing every crawler, this one included")
        if self.api_calls:
            best = self.api_calls[0]
            return ("json api", f"the site's own frontend calls {best['url'][:90]}")
        if self.has_job_posting_markup and self.job_sitemaps:
            return ("sitemap + json-ld",
                    "job URLs are enumerable from the sitemap and each page "
                    "carries schema.org JobPosting markup -- structured data, "
                    "no HTML parsing")
        if self.has_job_posting_markup:
            return ("json-ld", "pages carry schema.org JobPosting markup")
        if self.feeds:
            return ("rss", f"feed at {self.feeds[0]}")
        if self.job_sitemaps or self.sitemaps:
            return ("sitemap + html", "job URLs enumerable, but content needs parsing")
        if self.challenge:
            return ("blocked", "challenge wall; needs a commercial unblocker")
        return ("html", "no structured route found -- parsing required")


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def named_blocks(robots_text: str) -> list[str]:
    """Agents from NAMED_AGENTS that this robots.txt names and refuses.

    robots.txt permits STACKED user-agent lines -- several agents listed one
    after another, sharing the rule group that follows. Indeed does exactly
    that, listing ClaudeBot and anthropic-ai among a dozen others before a
    single Disallow.

    The first version of this assumed one agent per group and so captured
    nothing for a stacked block: it reported Indeed's job paths as ALLOWED when
    Indeed had refused this crawler by name. A safety check that fails open is
    worse than no check, because it is the one you stop verifying.
    """
    wanted = {agent.lower() for agent in NAMED_AGENTS}
    found: list[str] = []
    current_group: list[str] = []
    group_has_disallow = False

    def flush() -> None:
        if group_has_disallow:
            for agent in current_group:
                if agent.lower() in wanted and agent not in found:
                    found.append(agent)

    previous_was_agent = False
    for raw in robots_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()

        if key == "user-agent":
            # A user-agent line after any rule starts a NEW group.
            if not previous_was_agent and current_group:
                flush()
                current_group, group_has_disallow = [], False
            current_group.append(value)
            previous_was_agent = True
            continue

        previous_was_agent = False
        if key == "disallow" and value:
            group_has_disallow = True

    flush()
    return found


def check_robots(url: str) -> dict[str, Any]:
    """Read robots.txt. What the site permits, in its own words.

    Checked first and treated as decisive. Everything else this module does is
    only appropriate if this says so.
    """
    host = _host(url)
    base = f"{urllib.parse.urlparse(url).scheme}://{host}"
    result = net.fetch(f"{base}/robots.txt", accept="text/plain", cache_ttl_s=86400.0)

    out: dict[str, Any] = {"fetched": result.ok, "named_blocks": [],
                           "sitemaps": [], "allowed": None, "raw": ""}
    if not result.ok:
        return out

    text = result.text()
    out["raw"] = text[:4000]
    out["sitemaps"] = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", text)

    out["named_blocks"] = named_blocks(text)

    parser = urllib.robotparser.RobotFileParser()
    parser.parse(text.splitlines())
    for path in ("/jobs", "/job", "/careers", "/vacancies"):
        if parser.can_fetch(OUR_AGENT, urllib.parse.urljoin(base, path)):
            out["allowed"] = True
            break
    else:
        out["allowed"] = False
    return out


JOB_PATH_RE = re.compile(r"job|vacanc|position|opportunit|career", re.I)


def _looks_like_job_url(url: str) -> bool:
    """Does the PATH look like a job listing, ignoring the hostname?

    Matching the whole URL reported careers.gov.sg as having 20 job sitemaps --
    every entry matched, because the hostname contains 'careers'. It has 22
    static pages and no job URLs at all. A site's own name says nothing about
    what a given page is.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    if not path or path in ("/", ""):
        return False
    # A real job URL has an identifier in it, not just a section name.
    has_identifier = bool(re.search(r"\d{3,}|[0-9a-f]{8,}|-[a-z0-9]{6,}$", path, re.I))
    return bool(JOB_PATH_RE.search(path)) and has_identifier


def find_job_sitemaps(sitemaps: list[str], limit: int = 3) -> list[str]:
    """Follow sitemap indexes to the ones that actually list job pages."""
    found: list[str] = []
    for sitemap in sitemaps[:limit]:
        result = net.fetch(sitemap, accept="application/xml", cache_ttl_s=86400.0)
        if not result.ok:
            continue
        body = result.text()[:400000]
        children = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)

        job_urls = [c for c in children if _looks_like_job_url(c)]
        if job_urls:
            found.extend(job_urls)
            continue
        # A sitemap index points at more sitemaps; follow those whose own path
        # mentions jobs, which is how the big boards shard them.
        nested = [c for c in children
                  if c.endswith((".xml", ".xml.gz")) and JOB_PATH_RE.search(c)]
        found.extend(nested[:5])
    return found[:50]


def inspect_page(url: str, timeout: int = 40) -> dict[str, Any]:
    """Load a page in a browser and record what it fetches and embeds.

    The XHR capture is the part that matters. A site's own frontend has to get
    the listings from somewhere, and watching it ask is faster and far more
    reliable than reading minified bundles by hand.
    """
    out: dict[str, Any] = {"api_calls": [], "jsonld_types": [],
                           "has_job_posting": False, "feeds": [],
                           "challenge": False, "error": ""}
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        out["error"] = "playwright not installed"
        return out

    captured: list[dict[str, Any]] = []
    dropped: list[str] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(locale="en-SG",
                                              timezone_id="Asia/Singapore")
                page = context.new_page()

                # Capturing XHR is the whole reason this function drives a
                # browser. Swallowing a failure here silently produced the one
                # answer you must never trust: "this site exposes no JSON API",
                # reported with the same confidence whether none was called or
                # the listener broke on every response.
                def on_response(response):
                    try:
                        ctype = (response.headers or {}).get("content-type", "")
                        if "json" not in ctype.lower():
                            return
                        if response.request.resource_type not in ("xhr", "fetch"):
                            return
                        captured.append({
                            "url": response.url,
                            "method": response.request.method,
                            "status": response.status,
                            "post_data": (response.request.post_data or "")[:400],
                        })
                    except Exception as exc:
                        # Broad on purpose: Playwright raises its own Error type
                        # for a response whose body was discarded before this
                        # ran, and importing it here would couple recon to a
                        # module it only optionally depends on. Recorded, not
                        # swallowed.
                        dropped.append(f"{type(exc).__name__}: {exc}"[:100])

                page.on("response", on_response)
                page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
                page.wait_for_timeout(4000)   # let the app fetch its data
                html = page.content()
            finally:
                browser.close()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return out

    lowered = html[:60000].lower()
    out["challenge"] = any(m in lowered for m in
                           ("just a moment", "cf_chl", "captcha", "access denied"))

    for block in JSONLD_RE.findall(html):
        try:
            data = json.loads(block.strip())
        except ValueError:
            continue
        for item in (data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                continue
            kind = item.get("@type") or ""
            kinds = kind if isinstance(kind, list) else [kind]
            for k in kinds:
                if k and k not in out["jsonld_types"]:
                    out["jsonld_types"].append(k)
                if k == "JobPosting":
                    out["has_job_posting"] = True

    out["feeds"] = FEED_RE.findall(html)[:5]

    # Rank captured calls by how much they look like a job search endpoint.
    #
    # Scored on host and path only. Including the query string ranked a
    # LinkedIn ad pixel top on Tech in Asia, because the page URL it was
    # reporting sat inside its parameters and contained the word "jobs".
    # Analytics beacons quote the page they fire from, so a query-string match
    # says nothing about the endpoint itself.
    site_host = _host(url)

    def score(call: dict[str, Any]) -> int:
        parsed = urllib.parse.urlparse(call["url"])
        host, path = parsed.netloc.lower(), parsed.path.lower()

        if any(tracker in host for tracker in THIRD_PARTY_HOSTS):
            return -99
        points = sum(3 for word in ("job", "vacanc", "position", "search",
                                    "listing", "posting") if word in path)
        points += 2 if "/api/" in path or "graphql" in path else 0
        points += 2 if host.endswith(site_host.split(".", 1)[-1]) else 0
        points += 1 if call["status"] == 200 else -5
        return points

    ranked = sorted(captured, key=score, reverse=True)
    out["api_calls"] = [c for c in ranked if score(c) > 3][:6]
    if dropped:
        # Surfaced so "no JSON API found" can be distinguished from "the
        # listener could not read the responses that went past".
        out["error"] = (f"{len(dropped)} response(s) could not be inspected "
                        f"({dropped[0]}); the API-call list may be incomplete")
    return out


def recon(url: str, inspect: bool = True) -> Recon:
    """Full reconnaissance of one site. Never raises."""
    report = Recon(url=url, host=_host(url))

    robots = check_robots(url)
    report.robots_fetched = robots["fetched"]
    report.jobs_path_allowed = robots["allowed"]
    report.named_blocks = robots["named_blocks"]
    report.sitemaps = robots["sitemaps"]

    if report.named_blocks:
        # Context, not a verdict. These name other crawlers; what governs this
        # tool is the wildcard group checked above.
        report.notes.append(
            f"robots.txt separately refuses {', '.join(report.named_blocks)} -- "
            "those are other people's crawlers and do not describe this tool, "
            "but they tell you how the site feels about automated collection.")

    if report.jobs_path_allowed is False:
        report.notes.append(
            "robots.txt `User-agent: *` disallows the job paths; stopping.")
        return report

    if report.sitemaps:
        report.job_sitemaps = find_job_sitemaps(report.sitemaps)

    if inspect:
        page = inspect_page(url)
        report.api_calls = page["api_calls"]
        report.jsonld_types = page["jsonld_types"]
        report.has_job_posting_markup = page["has_job_posting"]
        report.feeds = page["feeds"]
        report.challenge = page["challenge"]
        if page["error"]:
            report.notes.append(page["error"])

    return report


def print_report(report: Recon) -> None:
    strategy, why = report.recommendation()
    print(f"\n=== {report.host} ===")
    print(f"  robots.txt        {'fetched' if report.robots_fetched else 'MISSING'}")
    if report.named_blocks:
        print(f"  NAMED BLOCKS      {', '.join(report.named_blocks)}")
    print(f"  job paths allowed {report.jobs_path_allowed}")
    print(f"  sitemaps          {len(report.sitemaps)} declared, "
          f"{len(report.job_sitemaps)} job-shaped")
    if report.job_sitemaps:
        print(f"                    {report.job_sitemaps[0][:80]}")
    print(f"  JSON-LD types     {', '.join(report.jsonld_types[:6]) or 'none'}")
    print(f"  JobPosting markup {report.has_job_posting_markup}")
    print(f"  RSS/Atom feeds    {len(report.feeds)}")
    print(f"  frontend JSON API {len(report.api_calls)} call(s)")
    for call in report.api_calls[:3]:
        print(f"                    {call['method']} {call['url'][:88]}")
        if call["post_data"]:
            print(f"                      body: {call['post_data'][:70]}")
    if report.challenge:
        print("  CHALLENGE WALL    yes")
    for note in report.notes:
        print(f"  note              {note}")
    print(f"\n  -> {strategy.upper()}: {why}\n")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for url in sys.argv[1:]:
        if not url.startswith("http"):
            url = "https://" + url
        print_report(recon(url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
