#!/usr/bin/env python3
"""
Quick connectivity/authorization test for the DeepSeek API key stored in .env.

Runs two checks:
  1. Authorization  -> GET  /user/balance        (free, no tokens spent)
  2. Inference      -> POST /chat/completions     (tiny prompt, ~1 line reply)

Usage:   py test_deepseek.py

Dependencies: none (Python standard library only).
"""

import json
import os
import sys
import urllib.error
import urllib.request

# Make emoji/UTF-8 output safe on Windows consoles (avoids UnicodeEncodeError).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"  # deepseek-chat / -reasoner are deprecated 2026-07-24
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_dotenv(path):
    """Minimal .env loader: KEY=VALUE lines; ignores comments/blank lines.
    Does not overwrite variables already present in the real environment."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _maybe_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def api_request(method, path, api_key, payload=None, timeout=30):
    """Return (status_code, parsed_body). parsed_body is a dict or a raw string."""
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


def check_authorization(api_key):
    print("1) Authorization check  ->  GET /user/balance")
    status, body = api_request("GET", "/user/balance", api_key)
    if status == 200:
        print("   ✅ Authorization successful — the key is valid.")
        if isinstance(body, dict):
            print(f"      is_available: {body.get('is_available')}")
            for info in body.get("balance_infos", []):
                print(f"      balance: {info.get('total_balance')} {info.get('currency')}")
        return True
    if status in (401, 403):
        print(f"   ❌ Authorization FAILED (HTTP {status}). "
              "The key is invalid, expired, or revoked.")
    else:
        print(f"   ❌ Unexpected response (HTTP {status}).")
    print(f"      response: {body}")
    return False


def check_inference(api_key):
    print(f"\n2) Inference check      ->  POST /chat/completions  (model: {MODEL})")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: DeepSeek OK"}],
        "max_tokens": 128,  # room for reasoning + the final answer
        "stream": False,
    }
    status, body = api_request("POST", "/chat/completions", api_key, payload)
    if status == 200 and isinstance(body, dict):
        try:
            choice = body["choices"][0]
            msg = choice.get("message", {})
            reply = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            if reply:
                print(f"   ✅ Model replied: {reply!r}")
            elif reasoning:
                print("   ✅ Model responded (reasoning only; final content empty).")
                print(f"      reasoning_content: {reasoning[:120]!r}")
            else:
                print(f"   ⚠️  200 OK but empty reply (finish_reason="
                      f"{choice.get('finish_reason')}).")
            print(f"      finish_reason: {choice.get('finish_reason')}")
            usage = body.get("usage") or {}
            if usage:
                print(f"      tokens — prompt: {usage.get('prompt_tokens')}, "
                      f"completion: {usage.get('completion_tokens')}")
            return True
        except (KeyError, IndexError, AttributeError):
            print(f"   ⚠️  Unexpected 200 response shape: {body}")
            return False
    print(f"   ❌ Inference failed (HTTP {status}).")
    print(f"      response: {body}")
    return False


def main():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("❌ DEEPSEEK_API_KEY not found.")
        print(f"   Expected it in {ENV_PATH} or the environment.")
        print("   Copy .env.example to .env and set your real key.")
        sys.exit(1)

    masked = api_key[:6] + "…" + api_key[-4:] if len(api_key) > 12 else "set"
    print(f"Using DEEPSEEK_API_KEY = {masked}")
    print(f"Base URL              = {BASE_URL}\n")

    if not check_authorization(api_key):
        print("\nResult: authorization did NOT work. Fix the key and re-run.")
        sys.exit(1)

    check_inference(api_key)
    print("\nDone. Authorization works. ✅")


if __name__ == "__main__":
    main()
