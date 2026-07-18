"""
Quick connectivity/authorization test for the DeepSeek API key stored in .env.

Runs two checks:
  1. Authorization  -> GET  /user/balance        (free, no tokens spent)
  2. Inference      -> POST /chat/completions     (tiny prompt, ~1 line reply)

Usage:   py test_deepseek.py

The model is no longer hardcoded -- it comes from model_config.resolve("fast"),
which reads models.json and falls back safely if that file is missing or stale.

Dependencies: none (Python standard library only).
"""

import sys

import deepseek_common as common
import model_config

common.enable_utf8_stdout()


def check_authorization(api_key):
    print("1) Authorization check  ->  GET /user/balance")
    status, body = common.api_request("GET", "/user/balance", api_key)
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


def check_inference(api_key, model):
    print(f"\n2) Inference check      ->  POST /chat/completions  (model: {model})")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: DeepSeek OK"}],
        "max_tokens": 128,  # room for reasoning + the final answer
        "stream": False,
    }
    status, body = common.api_request("POST", "/chat/completions", api_key, payload)
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
                details = usage.get("completion_tokens_details") or {}
                print(f"      tokens — prompt: {usage.get('prompt_tokens')}, "
                      f"completion: {usage.get('completion_tokens')}"
                      + (f", reasoning: {details.get('reasoning_tokens')}"
                         if details.get("reasoning_tokens") is not None else ""))
                cost = model_config.estimate_cost(
                    model, usage.get("prompt_tokens") or 0,
                    usage.get("completion_tokens") or 0)
                if cost is not None:
                    print(f"      est. cost: ${cost:.8f}")
            return True
        except (KeyError, IndexError, AttributeError):
            print(f"   ⚠️  Unexpected 200 response shape: {body}")
            return False
    print(f"   ❌ Inference failed (HTTP {status}).")
    print(f"      response: {body}")
    return False


def main():
    api_key = common.get_api_key()
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found.")
        print(f"   Expected it in {common.ENV_PATH} or the environment.")
        print("   Copy .env.example to .env and set your real key.")
        sys.exit(1)

    resolved = model_config.resolve_verbose("fast")
    masked = api_key[:6] + "…" + api_key[-4:] if len(api_key) > 12 else "set"
    print(f"Using DEEPSEEK_API_KEY = {masked}")
    print(f"Base URL              = {common.BASE_URL}")
    print(f"Model (tier 'fast')   = {resolved['model']}  [source: {resolved['source']}]")
    for warning in resolved["warnings"]:
        print(f"  ! {warning}")
    print()

    if not check_authorization(api_key):
        print("\nResult: authorization did NOT work. Fix the key and re-run.")
        sys.exit(1)

    check_inference(api_key, resolved["model"])
    print("\nDone. Authorization works. ✅")


if __name__ == "__main__":
    main()
