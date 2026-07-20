"""Shared plumbing for the DeepSeek helpers in this repo.

Extracted from test_deepseek.py so the updater, the resolver and the client all
use one copy of the .env loading and HTTP code. Standard library only.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# Overridable so the verification steps can point at a bad host to prove the
# failure paths degrade rather than crash.
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# src/jobbuddy/deepseek/<file> -> four levels up is the repo root.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
ENV_PATH = os.path.join(REPO_DIR, ".env")

USER_AGENT = "job-buddy/1.0 (+https://github.com/kimsiang1994/job-buddy)"


def enable_utf8_stdout():
    """Make UTF-8 output safe on Windows consoles (avoids UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def load_dotenv(path=ENV_PATH):
    """Minimal .env loader: KEY=VALUE lines; ignores comments and blank lines.

    Uses setdefault so a real environment variable always beats the file. That is
    what lets CI inject DEEPSEEK_API_KEY with no .env present, and what lets the
    verification steps override a value for one command.
    """
    if not os.path.exists(path):
        return
    # utf-8-sig, not utf-8: a .env written by Notepad or `Set-Content` carries a
    # BOM, and reading it as plain utf-8 makes the first key '﻿DEEPSEEK_API_KEY'
    # -- so the API key is silently absent and every call fails unauthenticated.
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key():
    """Return the API key, or None if missing or still the placeholder."""
    load_dotenv()
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key or key == "your_deepseek_api_key_here":
        return None
    return key


def _maybe_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def api_request(method, path, api_key, payload=None, timeout=30):
    """Call the DeepSeek API. Returns (status_code, parsed_body).

    parsed_body is a dict/list for JSON responses, else the raw string. Network
    failures come back as (0, "network error: ...") rather than raising, so every
    caller can treat the outcome as a status check instead of juggling exceptions.

    `timeout` defaults to 30s, which suits the non-thinking profiles this was
    written for. It is far too short for a reasoning model: `analyze` runs at
    reasoning_effort "high" and routinely spends longer than that thinking
    before emitting a first token. Callers using a thinking profile must pass a
    larger value -- `deepseek_client` derives one from the plan. Left at the
    default, every quality-tier call fails as a socket timeout, which then
    looks like a network fault rather than a configuration one.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _maybe_json(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, _maybe_json(err.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError) as err:
        return 0, f"network error: {err}"


def fetch_text(url, timeout=30):
    """GET a public URL and return its decoded body. Raises on failure.

    Deliberately separate from api_request(): the docs pages need no auth, which
    is what lets the CI docs-watch run without a key at all.
    """
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "text/html,application/xhtml+xml")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")
