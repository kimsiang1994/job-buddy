"""How a page gets fetched. Three strategies behind one interface.

Job sites vary enormously in how hard they are to read, and the right tool
differs per site. Measured on 2026-07-19 against seven Singapore job sites:

    plain HTTP      MyCareersFuture, Workable, Greenhouse, Lever, Ashby,
                    SmartRecruiters, Workday, HN -- all 200, all keyless
    browser         Tech in Asia -- 403 to curl, 200 to a stock headless
                    Chromium with no evasion. The 403 was missing headers, not
                    policy.
    unblocker       Glints, NodeFlair, JobStreet, FastJobs, Indeed, Glassdoor --
                    Cloudflare `cf_chl` challenge or CAPTCHA even for a real
                    browser. These refuse automation deliberately.

The escalation is deliberate and ordered by cost: try HTTP, then a browser,
then a paid service. Never start at the expensive end.

On the third tier: getting past an active challenge means fingerprint spoofing,
residential proxy rotation or CAPTCHA solving. This module does not implement
any of that. It delegates to a commercial unblocker you supply a key for --
those vendors operate the evasion as a service and carry their own compliance
position, which is a very different thing from a personal script doing it. If
you have not configured one, those sites are simply reported as unavailable
rather than half-attempted.

Configure via .env:

    SCRAPING_PROVIDER=scrapingbee     # or brightdata, zyte, scraperapi
    SCRAPING_API_KEY=...
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Any

from jobbuddy import net

# Sites measured as serving a plain headless browser with no evasion.
BROWSER_OK = {
    "www.techinasia.com", "techinasia.com",
}

# Sites that challenge even a real browser. Reachable only via a paid unblocker.
NEEDS_UNBLOCKER = {
    "glints.com", "www.glints.com",
    "nodeflair.com", "www.nodeflair.com",
    "sg.jobstreet.com", "www.jobstreet.com",
    "www.fastjobs.sg", "fastjobs.sg",
    "sg.indeed.com", "www.indeed.com",
    "www.glassdoor.sg", "www.glassdoor.com",
}

# Never attempted, at any tier, and not because of a technical obstacle.
#
# LinkedIn's User Agreement 8.2 prohibits automated access, and the account
# that gets restricted is the job seeker's own -- losing your LinkedIn
# mid-search is worse than missing its listings.
#
# The rest name this crawler family in their robots.txt: verified with
# site_recon, Indeed and NodeFlair list `ClaudeBot` and `anthropic-ai` by name
# in a disallow group, while allowing LinkedInBot and Googlebot. A generic bot
# wall is a site defending itself against load; naming an agent is a site
# answering a question it was asked. Routing an unblocker around that would be
# overriding an explicit answer, so these are excluded before the unblocker
# tier is ever reached. Their inventory is reachable through the aggregator
# APIs instead, where a vendor has its own relationship with the source.
NEVER = {
    "www.linkedin.com", "linkedin.com",
    "sg.indeed.com", "www.indeed.com", "indeed.com",
    "nodeflair.com", "www.nodeflair.com",
    "sg.jobstreet.com", "www.jobstreet.com", "jobstreet.com",
    "hk.jobsdb.com", "sg.jobsdb.com",
    "sg.jora.com", "jora.com",
}

UNBLOCKER_ENDPOINTS = {
    "scrapingbee": "https://app.scrapingbee.com/api/v1/",
    "scraperapi": "https://api.scraperapi.com/",
    "zyte": "https://api.zyte.com/v1/extract",
    "brightdata": "https://api.brightdata.com/request",
}


@dataclass
class PageResult:
    """One fetched page, however it was obtained."""

    ok: bool
    url: str
    html: str = ""
    status: int = 0
    strategy: str = ""
    error: str = ""
    cost_note: str = ""


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def strategy_for(url: str) -> str:
    """Which tier this host needs. Cheapest that will work."""
    host = _host(url)
    if host in NEVER:
        return "never"
    if host in NEEDS_UNBLOCKER:
        return "unblocker"
    if host in BROWSER_OK:
        return "browser"
    return "http"


def unblocker_configured() -> tuple[str, str] | None:
    """(provider, key) if one is set up, else None."""
    provider = (os.environ.get("SCRAPING_PROVIDER") or "").strip().lower()
    key = (os.environ.get("SCRAPING_API_KEY") or "").strip()
    if provider in UNBLOCKER_ENDPOINTS and key:
        return provider, key
    return None


def fetch_page(url: str, timeout: int = 40, cache_ttl_s: float = 3600.0) -> PageResult:
    """Fetch one page using the cheapest strategy that works for its host."""
    strategy = strategy_for(url)

    if strategy == "never":
        return PageResult(False, url, strategy="never",
                          error="host is excluded: automated access prohibited "
                                "and the user's own account is the collateral")

    if strategy == "http":
        result = net.fetch(url, accept="text/html,application/json", cache_ttl_s=cache_ttl_s)
        return PageResult(result.ok, result.url, result.text(), result.status,
                          "http", result.error)

    if strategy == "browser":
        return _fetch_browser(url, timeout)

    return _fetch_unblocker(url, timeout)


def _fetch_browser(url: str, timeout: int) -> PageResult:
    """Stock headless Chromium. No stealth plugin, no proxy, no patching.

    If a site needs more than this to serve a page, it is saying no, and the
    answer is a commercial unblocker rather than a cleverer script.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return PageResult(False, url, strategy="browser",
                          error="playwright not installed -- "
                                "py -m pip install playwright && py -m playwright install chromium")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    locale="en-SG", timezone_id="Asia/Singapore",
                    viewport={"width": 1366, "height": 900},
                )
                page = context.new_page()
                response = page.goto(url, timeout=timeout * 1000,
                                     wait_until="domcontentloaded")
                page.wait_for_timeout(2500)
                html = page.content()
                status = response.status if response else 0
            finally:
                browser.close()
    except Exception as exc:
        return PageResult(False, url, strategy="browser",
                          error=f"{type(exc).__name__}: {exc}"[:180])

    if _looks_challenged(html):
        return PageResult(False, url, html, status, "browser",
                          "challenge wall -- needs an unblocker")
    return PageResult(status < 400, url, html, status, "browser")


def _fetch_unblocker(url: str, timeout: int) -> PageResult:
    """Delegate to a commercial unblocking service, if one is configured."""
    configured = unblocker_configured()
    if not configured:
        return PageResult(
            False, url, strategy="unblocker",
            error="this host challenges browsers; set SCRAPING_PROVIDER and "
                  "SCRAPING_API_KEY in .env to reach it",
        )

    provider, key = configured
    if provider == "scrapingbee":
        target = (f"{UNBLOCKER_ENDPOINTS[provider]}?api_key={key}"
                  f"&url={urllib.parse.quote(url, safe='')}"
                  f"&render_js=true&premium_proxy=true&country_code=sg")
        result = net.fetch(target, timeout=timeout, accept="text/html")
    elif provider == "scraperapi":
        target = (f"{UNBLOCKER_ENDPOINTS[provider]}?api_key={key}"
                  f"&url={urllib.parse.quote(url, safe='')}"
                  f"&render=true&country_code=sg")
        result = net.fetch(target, timeout=timeout, accept="text/html")
    elif provider == "zyte":
        # Zyte takes a POST body and HTTP Basic auth on the API key.
        import base64

        auth = base64.b64encode(f"{key}:".encode()).decode()
        result = net.fetch(
            UNBLOCKER_ENDPOINTS[provider], method="POST",
            payload={"url": url, "browserHtml": True,
                     "geolocation": "SG"},
            headers={"Authorization": f"Basic {auth}"},
            timeout=timeout, accept="application/json",
        )
        if result.ok:
            data = result.json() or {}
            return PageResult(True, url, data.get("browserHtml", ""),
                              result.status, f"unblocker:{provider}",
                              cost_note="billed per request")
    else:  # brightdata
        result = net.fetch(
            UNBLOCKER_ENDPOINTS[provider], method="POST",
            payload={"zone": os.environ.get("BRIGHTDATA_ZONE", "web_unlocker"),
                     "url": url, "format": "raw", "country": "sg"},
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout, accept="text/html",
        )

    return PageResult(result.ok, url, result.text(), result.status,
                      f"unblocker:{provider}", result.error,
                      cost_note="billed per request")


_CHALLENGE_MARKERS = (
    "just a moment", "cf_chl", "cf-challenge", "checking your browser",
    "captcha", "verify you are human", "enable javascript and cookies",
    "access denied", "request blocked",
)


def _looks_challenged(html: str) -> bool:
    """True when the response is a bot-challenge page rather than content."""
    lowered = (html or "")[:60000].lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


def availability_report() -> list[dict[str, Any]]:
    """What is reachable right now, and what it would take. For the notebook."""
    configured = unblocker_configured()
    rows = [
        {"tier": "http", "hosts": "MyCareersFuture, Workable, Greenhouse, Lever, "
                                  "Ashby, SmartRecruiters, Workday, HN",
         "available": True, "needs": "nothing -- keyless"},
        {"tier": "browser", "hosts": ", ".join(sorted(BROWSER_OK)),
         "available": _playwright_present(), "needs": "playwright + chromium"},
        {"tier": "unblocker", "hosts": ", ".join(sorted(NEEDS_UNBLOCKER)),
         "available": bool(configured),
         "needs": (f"configured: {configured[0]}" if configured
                   else "SCRAPING_PROVIDER + SCRAPING_API_KEY in .env")},
        {"tier": "excluded", "hosts": ", ".join(sorted(NEVER)),
         "available": False,
         "needs": "not attempted -- your own account is the collateral"},
    ]
    return rows


def _playwright_present() -> bool:
    try:
        import playwright  # noqa: F401

        return True
    except ImportError:
        return False
