"""Runs one tactic against another and reports whether it actually helped.

    py -m jobbuddy.experiment --arms baseline,xyz_formula --trials 30

This is the thing that stops the project accumulating folklore. Resume advice
is almost entirely untested assertion, and the obvious way to test a tactic --
tailor twice, score both, compare -- produces confident nonsense, because an
LLM grader scoring an UNCHANGED resume 100 times spans a range wider than any
tactic's real effect. So an arm here is never judged on a single score.

The pipeline per arm:

    profile + job -> tailor(strategy_names=arm) -> gated bullets -> plain text

and then `ab_harness` does the comparing: a noise floor first, paired forced
choice in both orderings, bootstrap CI, and a verdict that is allowed to be
"no detectable difference" -- which is the expected answer for most tactics and
is a result rather than a failure.

**The control arm is not optional.** `baseline` is always included even when
not named, because a comparison between two tactics with no control cannot
tell you that both made things worse.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jobbuddy import ab_harness, hr_panel, strategies, tailor

REPO_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_DIR / "experiments"


def as_text(profile: dict[str, Any], tailored: dict[str, Any]) -> str:
    """Flatten a tailored draft into the plain text a screener would read.

    Deliberately plain: the graders compare CONTENT, and handing them layout
    invites them to score typography instead. Layout is measured separately and
    deterministically by `render_resume.page_one_sufficiency`.
    """
    identity = profile.get("identity") or {}
    lines = [str(identity.get("name") or "Candidate")]
    headline = (tailored.get("headline") or "").strip()
    if headline:
        lines.append(headline)
    lines.append("")

    current_role = None
    for bullet in tailored.get("bullets") or []:
        role = f"{bullet.get('role') or ''} — {bullet.get('org') or ''}".strip(" —")
        if role and role != current_role:
            lines.append("")
            lines.append(role)
            current_role = role
        lines.append(f"- {bullet.get('text') or ''}")
    return "\n".join(lines).strip()


def build_arms(profile: dict[str, Any], job: dict[str, Any],
               requirements: list[str], arm_names: list[str],
               chat: Callable[..., dict[str, Any]] | None = None,
               equalise_length: bool = True
               ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Tailor once per arm. Returns ({arm: resume_text}, per-arm detail).

    `equalise_length` truncates every arm to the shortest arm's bullet count,
    and it defaults to ON because of a result this runner produced on its first
    live run. `xyz_formula` beat `baseline` 0.92 to 0.08 with a bootstrap
    interval nowhere near chance -- an emphatic, entirely uninterpretable
    result, because that arm had emitted 8 bullets against baseline's 3. The
    graders were shown a fuller resume and a thinner one and asked which was
    stronger. Nothing about XYZ structure was being measured.

    A tactic that changes how many bullets survive is a real effect, but it is
    a DIFFERENT effect from the one the tactic claims, and the two are
    inseparable unless length is held fixed. So arms are equalised by default,
    and the pre-truncation counts are reported so a length difference is
    visible rather than silently removed.

    Turn it off only to measure a tactic's effect on yield deliberately, and
    read the result as "this arm produced more" rather than "this arm is
    better".
    """
    # The control is added rather than assumed. Comparing two tactics without
    # it cannot distinguish "A beats B" from "both are worse than doing
    # nothing", and the second is a real possibility worth detecting.
    if "baseline" not in arm_names:
        arm_names = ["baseline"] + arm_names

    drafts: dict[str, dict[str, Any]] = {}
    detail: list[dict[str, Any]] = []

    for name in arm_names:
        names = None if name == "baseline" else sorted(set(strategies.defaults()) | {name})
        drafted = tailor.tailor(profile, job, requirements, chat=chat,
                                strategy_names=names)
        if not drafted.get("ok"):
            detail.append({"arm": name, "ok": False, "error": drafted.get("error")})
            continue
        if not (drafted.get("bullets") or []):
            detail.append({"arm": name, "ok": False,
                           "error": "arm produced no bullets"})
            continue
        drafts[name] = drafted

    if not drafts:
        return {}, detail

    produced = {name: len(d["bullets"]) for name, d in drafts.items()}
    limit = min(produced.values()) if equalise_length else None

    variants: dict[str, str] = {}
    for name, drafted in drafts.items():
        shown = drafted if limit is None else {
            **drafted, "bullets": drafted["bullets"][:limit]}
        text = as_text(profile, shown)
        if not text.strip():
            detail.append({"arm": name, "ok": False, "error": "no text after truncation"})
            continue
        variants[name] = text
        detail.append({
            "arm": name, "ok": True,
            "bullets_produced": produced[name],
            "bullets_shown": len(shown["bullets"]),
            "fell_back": sum(1 for b in shown["bullets"] if b.get("fell_back")),
            "guard_rejected": (drafted.get("guard") or {}).get("rejected"),
            "unaddressed": len(drafted.get("unaddressed") or []),
            "strategies": drafted.get("strategies"),
            "cost_usd": drafted.get("cost_usd"),
        })

    if limit is not None and len(set(produced.values())) > 1:
        detail.append({
            "arm": "_note", "ok": True,
            "length_control": (
                f"arms produced {produced}; all truncated to {limit} bullets so "
                "the comparison measures the tactic rather than resume length"),
        })
    return variants, detail


def run(profile: dict[str, Any], job: dict[str, Any],
        requirements: list[str] | None = None,
        arm_names: list[str] | None = None,
        trials: int = 30,
        chat: Callable[..., dict[str, Any]] | None = None,
        seed: int | str = 0,
        max_calls: int = 500,
        equalise_length: bool = True) -> dict[str, Any]:
    """Run the experiment end to end. Never raises."""
    requirements = requirements or []
    arm_names = arm_names or ["baseline"]

    variants, detail = build_arms(profile, job, requirements, arm_names, chat,
                                  equalise_length=equalise_length)
    if len(variants) < 2:
        return {"ok": False,
                "error": f"need 2+ working arms, got {len(variants)}",
                "arms": detail}

    personas = hr_panel.build_personas(job, requirements)
    planned = ab_harness.estimate_cost(variants, trials, personas)

    # The floor is measured on the control, on the same job, with the same
    # panel -- a floor borrowed from a different job would not describe this
    # comparison's null distribution.
    floor = ab_harness.noise_floor(variants["baseline"], job, requirements,
                                   trials=max(5, trials // 3), chat=chat)

    comparison = ab_harness.compare_variants(
        variants, job, requirements, floor,
        trials=trials, chat=chat, personas=personas,
        seed=seed, max_calls=max_calls)

    return {
        "ok": True,
        "job": {"title": job.get("title"), "company": job.get("company")},
        "arms": detail,
        "planned_calls": planned,
        "noise_floor": floor,
        "comparison": comparison,
        "summary": ab_harness.summarise(comparison),
    }


def save(result: dict[str, Any], directory: Path | None = None) -> Path:
    """Write the result, timestamped. Results accumulate; none are overwritten."""
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"experiment-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


def _report(result: dict[str, Any]) -> None:
    if not result.get("ok"):
        print(f"experiment failed: {result.get('error')}", file=sys.stderr)
        for arm in result.get("arms") or []:
            if not arm.get("ok"):
                print(f"  {arm.get('arm')}: {arm.get('error')}", file=sys.stderr)
        return

    job = result["job"]
    print(f"\njob: {job['title']} @ {job['company']}\n")

    print("arms")
    for arm in result["arms"]:
        if arm.get("length_control"):
            print(f"  note: {arm['length_control']}")
        elif arm.get("ok"):
            print(f"  {arm['arm']:22} {arm['bullets_shown']:2d} shown "
                  f"({arm['bullets_produced']} produced)  "
                  f"{arm['fell_back']} fell back  "
                  f"{arm['unaddressed']} unaddressed")
        else:
            print(f"  {arm['arm']:22} FAILED: {arm.get('error')}")

    floor = result["noise_floor"]
    print(f"\nnoise floor (identical input, {floor.get('trials')} runs)")
    print(f"  mean {floor.get('mean')}  sd {floor.get('sd')}  "
          f"spread {floor.get('min')}-{floor.get('max')}")
    print("  any difference smaller than this is noise, not a finding")

    comparison = result["comparison"]
    print(f"\nverdict: {comparison.get('verdict')}")
    for name, row in (comparison.get("variants") or {}).items():
        flags = []
        if row.get("significant"):
            flags.append("SIGNIFICANT")
        if row.get("underpowered"):
            flags.append(f"underpowered (needs ~{row.get('trials_needed')})")
        print(f"  {name:22} win {row.get('win_rate')}  "
              f"CI [{row.get('ci_low')}, {row.get('ci_high')}]  "
              f"n={row.get('comparisons')}  {' '.join(flags)}")
    print(f"\n  {comparison.get('reason')}")
    if comparison.get("failures"):
        print(f"  {len(comparison['failures'])} grader failure(s)")


def main() -> int:
    from jobbuddy.deepseek import deepseek_common
    deepseek_common.load_dotenv()

    from jobbuddy import job_store, pipeline, scoring, verify_profile

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default="baseline,xyz_formula",
                        help="comma-separated strategy names; baseline is always added")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--no-length-control", action="store_true",
                        help="do NOT equalise bullet counts across arms; the "
                             "result then measures yield, not the tactic")
    args = parser.parse_args()

    profile_path = REPO_DIR / "profile" / "master_profile.json"
    if not profile_path.is_file():
        print("no verified profile -- run import_resume then verify_profile.promote",
              file=sys.stderr)
        return 2
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    config = scoring.load_config()
    scopes = config.get("scopes") or []
    if not scopes:
        print("run_config.json defines no scopes", file=sys.stderr)
        return 2

    result_set = pipeline.run_scopes([scopes[0]], config, limit=8,
                                     cache_ttl_s=3600.0, dry_run=True,
                                     history=job_store.JobHistory.load())
    if not result_set.jobs:
        print("no jobs available to experiment on", file=sys.stderr)
        return 2

    job = result_set.jobs[min(args.job_index, len(result_set.jobs) - 1)]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    result = run(profile, job, (job.get("skills") or [])[:10], arms,
                 trials=args.trials, seed=args.seed, max_calls=args.max_calls,
                 equalise_length=not args.no_length_control)
    _report(result)
    if result.get("ok"):
        print(f"\nwrote {save(result)}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
