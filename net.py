"""Hardened HTTP for the public web.

`deepseek_common.api_request` talks to one API we trust. This module talks to
arbitrary job boards, which is a different threat model:

  - a job description page can be 50 MB, or an infinite redirect loop, or HTML
    when you asked for JSON;
  - hammering a free public endpoint gets your IP blocked, and MCF/GDELT/GitHub
    all publish different rate limits;
  - a run that re-fetches the same page 40 times is slow *and* rude.

So: size caps, redirect caps, content-type checks, a per-host token bucket, and
an on-disk cache. Same contract as the rest of the repo -- **this never raises**.
Callers branch on `FetchResult.ok`.

Deliberately stdlib-only (urllib). The core pipeline must run with zero installed
packages; only rendering is allowed to need a wheel.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_DIR / "state" / "http_cache"

USER_AGENT = "job-buddy/1.0 (+https://github.com/kimsiang1994/job-buddy)"

MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
DEFAULT_TIMEOUT = 30

# Retry only what a retry can actually fix. 4xx other than 408/429 means the
# request itself is wrong -- retrying hides the bug and wastes the quota.
RETRY_STATUSES = frozenset({0, 408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4

# Per-host minimum gap between requests, seconds. These are not guesses:
# GDELT documents 1 req/5s and returns 429 below it; MCF and the CKAN endpoints
# are government services where being a good citizen is the whole deal.
HOST_MIN_INTERVAL = {
    "api.gdeltproject.org": 5.0,
    "api.mycareersfuture.gov.sg": 0.35,
    "data.gov.sg": 0.5,
    "api.github.com": 0.8,
    "jobs.workable.com": 0.5,
    "boards-api.greenhouse.io": 0.35,
    "api.lever.co": 0.35,
    "api.ashbyhq.com": 0.35,
    "api.smartrecruiters.com": 0.35,
    "remotive.com": 30.0,  # they ask for ~4 fetches/day; treat as a hard brake
    "hn.algolia.com": 0.2,
}
DEFAULT_MIN_INTERVAL = 1.0

_ALLOWED_SCHEMES = ("http", "https")

_rate_lock = threading.Lock()
_last_request_at: dict[str, float] = {}
_warned: set[str] = set()


def _warn(message: str) -> None:
    """Warn once per distinct message, like model_config does."""
    if message in _warned:
        return
    _warned.add(message)
    try:
        import sys

        print(f"net: {message}", file=sys.stderr)
    except Exception:
        pass


@dataclass
class FetchResult:
    """Outcome of one fetch. Never an exception."""

    ok: bool
    status: int
    url: str
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    from_cache: bool = False
    error: str = ""
    attempts: int = 1
    elapsed_s: float = 0.0

    def text(self, encoding: str | None = None) -> str:
        """Decode the body, guessing the encoding from headers if not given.

        Job boards mislabel encodings constantly (MCF serves company blurbs with
        stray bytes), so this never raises -- it replaces bad bytes instead.
        """
        if encoding is None:
            ctype = self.headers.get("content-type", "")
            encoding = "utf-8"
            if "charset=" in ctype:
                encoding = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        try:
            return self.body.decode(encoding, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any | None:
        """Parse the body as JSON, or return None. Never raises."""
        if not self.body:
            return None
        try:
            return json.loads(self.text())
        except (ValueError, TypeError) as exc:
            _warn(f"{self.url}: body is not valid JSON ({exc})")
            return None


def _throttle(host: str) -> None:
    """Block until this host's minimum interval has elapsed."""
    interval = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
    while True:
        with _rate_lock:
            now = time.monotonic()
            last = _last_request_at.get(host, 0.0)
            wait = (last + interval) - now
            if wait <= 0:
                _last_request_at[host] = now
                return
        # Sleep outside the lock so other hosts are not blocked behind this one.
        time.sleep(min(wait, interval))


def _cache_path(method: str, url: str, body: bytes | None) -> Path:
    digest = hashlib.sha256(
        b"|".join([method.encode(), url.encode(), body or b""])
    ).hexdigest()
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def _read_cache(path: Path, ttl_s: float) -> FetchResult | None:
    if ttl_s <= 0 or not path.is_file():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl_s:
            return None
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return FetchResult(
            ok=True,
            status=int(raw["status"]),
            url=raw["url"],
            body=bytes.fromhex(raw["body_hex"]),
            headers=raw.get("headers", {}),
            from_cache=True,
        )
    except (OSError, ValueError, KeyError) as exc:
        _warn(f"cache read failed for {path.name} ({exc}); refetching")
        return None


def _write_cache(path: Path, result: FetchResult) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": result.status,
            "url": result.url,
            "headers": result.headers,
            "body_hex": result.body.hex(),
            "cached_at": time.time(),
        }
        # Atomic write, same as update_models.py: a half-written cache entry
        # that still parses is far worse than no cache entry.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        _warn(f"cache write failed for {path.name} ({exc}); continuing uncached")


def _decompress(raw: bytes, encoding: str) -> bytes:
    try:
        if encoding == "gzip":
            return gzip.decompress(raw)
        if encoding in ("deflate", "zlib"):
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    except (OSError, zlib.error) as exc:
        _warn(f"could not decompress {encoding} response ({exc}); using raw bytes")
    return raw


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stop urllib following redirects so we can count and inspect them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _read_capped(resp: Any) -> tuple[bytes, bool]:
    """Read at most MAX_BYTES. Returns (body, truncated)."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = resp.read(65536)
        if not chunk:
            return b"".join(chunks), False
        total += len(chunk)
        if total > MAX_BYTES:
            chunks.append(chunk[: MAX_BYTES - (total - len(chunk))])
            return b"".join(chunks), True
        chunks.append(chunk)


def fetch(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    cache_ttl_s: float = 0.0,
    accept: str = "application/json",
    expect_content_type: str | None = None,
) -> FetchResult:
    """Fetch a URL safely. Never raises.

    `payload` is JSON-encoded when it is not bytes -- Workday's CXS endpoint and
    Ashby's GraphQL board both need POST bodies.

    `cache_ttl_s > 0` serves an on-disk copy when one is fresh enough. Use it
    freely: re-running a pipeline during development should not re-hit a public
    endpoint dozens of times.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return FetchResult(False, 0, url, error=f"refusing non-http scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        return FetchResult(False, 0, url, error="url has no host")

    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    cache_file = _cache_path(method, url, body_bytes)
    cached = _read_cache(cache_file, cache_ttl_s)
    if cached is not None:
        return cached

    req_headers = {
        "User-Agent": USER_AGENT,
        "Accept": accept,
        "Accept-Encoding": "gzip, deflate",
    }
    if body_bytes is not None:
        req_headers["Content-Type"] = "application/json"
    if headers:
        req_headers.update(headers)

    opener = urllib.request.build_opener(_NoRedirect)
    started = time.monotonic()
    current_url = url
    redirects = 0
    attempt = 0
    last_error = ""
    last_status = 0

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        host = urllib.parse.urlparse(current_url).netloc
        _throttle(host)

        req = urllib.request.Request(
            current_url, data=body_bytes, headers=req_headers, method=method
        )
        try:
            with opener.open(req, timeout=timeout) as resp:
                status = resp.status
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                raw, truncated = _read_capped(resp)
        except urllib.error.HTTPError as exc:
            status = exc.code
            resp_headers = {k.lower(): v for k, v in (exc.headers or {}).items()}
            try:
                raw, truncated = _read_capped(exc)
            except Exception:
                raw, truncated = b"", False
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status, resp_headers, raw, truncated = 0, {}, b"", False
            last_error = f"network error: {exc}"

        # --- redirects: followed manually so we can cap them and, importantly,
        # so the caller can see the *final* url. An expired job posting usually
        # 30x's to /careers, and that destination is the evidence it is dead.
        if status in (301, 302, 303, 307, 308):
            location = resp_headers.get("location", "")
            if not location:
                return FetchResult(False, status, current_url, error="redirect without Location")
            redirects += 1
            if redirects > MAX_REDIRECTS:
                return FetchResult(
                    False, status, current_url,
                    error=f"exceeded {MAX_REDIRECTS} redirects",
                    attempts=attempt,
                    elapsed_s=time.monotonic() - started,
                )
            current_url = urllib.parse.urljoin(current_url, location)
            attempt -= 1  # a redirect hop is not a failed attempt
            continue

        if status in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            last_status = status
            if not last_error:
                last_error = f"HTTP {status}"
            # Honour Retry-After when the server bothered to tell us.
            delay = 2.0 ** attempt
            retry_after = resp_headers.get("retry-after", "")
            if retry_after.strip().isdigit():
                delay = max(delay, float(retry_after.strip()))
            time.sleep(min(delay, 30.0))
            continue

        if status == 0:
            return FetchResult(
                False, 0, current_url, error=last_error or "network error",
                attempts=attempt, elapsed_s=time.monotonic() - started,
            )

        body = _decompress(raw, resp_headers.get("content-encoding", "").lower())
        if truncated:
            _warn(f"{current_url}: response exceeded {MAX_BYTES} bytes and was truncated")

        ctype = resp_headers.get("content-type", "")
        if expect_content_type and expect_content_type not in ctype:
            # Workday returns HTML instead of JSON when the Accept header is
            # missing, and several boards serve an SPA shell where an API used
            # to be. Both look like success until you try to parse.
            return FetchResult(
                False, status, current_url, body=body, headers=resp_headers,
                error=f"expected content-type {expect_content_type!r}, got {ctype!r}",
                attempts=attempt, elapsed_s=time.monotonic() - started,
            )

        result = FetchResult(
            ok=200 <= status < 300,
            status=status,
            url=current_url,
            body=body,
            headers=resp_headers,
            error="" if 200 <= status < 300 else f"HTTP {status}",
            attempts=attempt,
            elapsed_s=time.monotonic() - started,
        )
        if result.ok and cache_ttl_s > 0 and not truncated:
            _write_cache(cache_file, result)
        return result

    return FetchResult(
        False, last_status, current_url,
        error=last_error or f"gave up after {MAX_ATTEMPTS} attempts",
        attempts=attempt, elapsed_s=time.monotonic() - started,
    )


def get_json(url: str, **kwargs: Any) -> tuple[Any | None, FetchResult]:
    """Fetch and parse JSON. Returns (data_or_None, result) so the caller can
    distinguish 'request failed' from 'request fine, body was not JSON'."""
    kwargs.setdefault("accept", "application/json")
    result = fetch(url, **kwargs)
    return (result.json() if result.ok else None), result


def clear_cache() -> int:
    """Delete the on-disk HTTP cache. Returns the file count removed."""
    if not CACHE_DIR.is_dir():
        return 0
    removed = 0
    for path in CACHE_DIR.rglob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
