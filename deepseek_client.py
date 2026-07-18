"""The single call path for DeepSeek chat completions.

    import deepseek_client
    result = deepseek_client.complete("Is this a job ad? yes/no", profile="classify")
    print(result["text"])

Responsibilities, in order:
  1. resolve the model from a capability tier   (model_config)
  2. resolve max_tokens + thinking from a task profile   (token_budget)
  3. call the API
  4. append the real usage to usage_log.jsonl  -- this is the ground truth that
     calibrate_budgets.py later tunes against, and the only way to measure how
     accurate the token estimator actually is
  5. retry once at a larger budget if the reply was truncated

Step 5 is what makes an estimation error self-correcting rather than a bug.
"""

import json
import os
import sys
from datetime import datetime, timezone

import deepseek_common as common
import model_config
import token_budget

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
USAGE_LOG = os.path.join(REPO_DIR, "usage_log.jsonl")

# Upper bound for the truncation retry, so a pathological case cannot escalate
# into a very expensive call.
RETRY_CEILING = 32768


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_usage(record):
    """Append one JSONL line. Logging must never break a call."""
    try:
        with open(USAGE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as err:
        print(f"[deepseek_client] could not write usage log: {err}", file=sys.stderr)


def _extract(body):
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "text": (message.get("content") or "").strip(),
        "reasoning": (message.get("reasoning_content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
    }


def chat(messages, profile=token_budget.DEFAULT_PROFILE, tier="fast", model=None,
         retry_on_truncation=True, **overrides):
    """Low-level entry point. -> result dict (never raises for API errors)."""
    api_key = common.get_api_key()
    if not api_key:
        return {"ok": False, "text": "", "error": "DEEPSEEK_API_KEY not set",
                "model": None, "profile": profile, "attempts": 0}

    plan = token_budget.budget_for(profile, messages, model, tier)
    # An explicit max_tokens becomes the STARTING budget rather than a payload
    # override -- otherwise it would be re-applied on the retry and the
    # truncation recovery below could never actually raise the ceiling.
    if "max_tokens" in overrides:
        max_tokens = int(overrides.pop("max_tokens"))
    else:
        max_tokens = plan["max_tokens"]
    attempts = 0
    retried = False

    while True:
        attempts += 1
        payload = {
            "model": plan["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if plan["thinking"]:
            payload["thinking"] = plan["thinking"]
        payload.update(overrides)

        status, body = common.api_request("POST", "/chat/completions",
                                          api_key, payload)

        if status != 200 or not isinstance(body, dict):
            return {"ok": False, "text": "", "reasoning": "",
                    "error": f"HTTP {status}: {str(body)[:300]}",
                    "model": plan["model"], "profile": plan["profile"],
                    "attempts": attempts, "retried": retried}

        parts = _extract(body)
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}

        _log_usage({
            "ts": _now(),
            "profile": plan["profile"],
            "model": plan["model"],
            "max_tokens": max_tokens,
            "thinking": (plan["thinking"] or {}).get("type"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "finish_reason": parts["finish_reason"],
            # Logged next to the real count so the estimator's error is measurable.
            "estimated_prompt_tokens": plan["estimated_prompt_tokens"],
            "estimator": token_budget.backend_name(),
            "attempt": attempts,
        })

        truncated = parts["finish_reason"] == "length"
        can_retry = (retry_on_truncation and truncated and attempts == 1
                     and max_tokens < RETRY_CEILING)
        if can_retry:
            # The estimate was too small -- double it once and try again.
            max_tokens = min(max_tokens * 2, RETRY_CEILING)
            retried = True
            continue

        cost = model_config.estimate_cost(
            plan["model"], usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0)

        return {
            "ok": bool(parts["text"]) or not truncated,
            "text": parts["text"],
            "reasoning": parts["reasoning"],
            "model": plan["model"],
            "profile": plan["profile"],
            "finish_reason": parts["finish_reason"],
            "usage": usage,
            "max_tokens": max_tokens,
            "attempts": attempts,
            "retried": retried,
            "truncated": truncated,
            "cost_usd": cost,
            "error": None,
        }


def complete(prompt, profile=token_budget.DEFAULT_PROFILE, tier="fast",
             system=None, model=None, retry_on_truncation=True, **overrides):
    """Convenience wrapper for a single-turn prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, profile=profile, tier=tier, model=model,
                retry_on_truncation=retry_on_truncation, **overrides)


if __name__ == "__main__":
    common.enable_utf8_stdout()
    which = sys.argv[1] if len(sys.argv) > 1 else "classify"
    demo = "Reply with exactly: DeepSeek OK"
    outcome = complete(demo, profile=which)
    print(json.dumps({k: v for k, v in outcome.items() if k != "reasoning"},
                     indent=2, ensure_ascii=False))
