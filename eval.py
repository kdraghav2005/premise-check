#!/usr/bin/env python3
"""Evaluate on data/test_cases.csv (7 dev cases, 5 holdout cases) and save every raw output.

  python eval.py                      run the 7 DEV cases (use these to tune the prompt)
  python eval.py --freeze             record the prompt + configuration + data hashes in freeze/frozen.json
  python eval.py --split holdout      run the 5 HOLDOUT cases; needs a matching freeze and runs once per freeze
  python eval.py --split all          dev + holdout (same freeze rule)

Only the pitch reaches the app. expected_outcome, reference_show_id and grading_note are read after the app answers.

Honest reporting:
  label agreement      the result's verdict equals the expected label (any status)
  verified success     label agrees AND the source checks passed (AND, for sequence cases, the intended show was named)
  An unverified answer can agree on the label but never counts as a verified success.
  Provider errors, unverified reports and tool errors are counted and shown, never hidden in a denominator.
  The keyword baseline does not produce the four verdicts; it is graded on retrieval of the intended show only.
A mock-provider run is a plumbing check, labelled as such, and is not model accuracy."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pitchcheck import prompts  # noqa: E402
from pitchcheck.llm import is_mock, load_config, public_config  # noqa: E402
from pitchcheck.pipeline import TOOL_VERSION, CardCache, check_pitch, load_catalog  # noqa: E402
from pitchcheck.safe import as_dict, as_list  # noqa: E402
from pitchcheck.shortlist import rank  # noqa: E402
from pitchcheck.textutil import norm, setup_stdout  # noqa: E402

OUTCOME_TO_VERDICT = {
    "sequence_overlap": "same_sequence",
    "shared_trope_only": "shared_trope_only",
    "no_meaningful_match": "no_meaningful_match",
    "insufficient_information": "insufficient_information",
}
REQUIRED = ("test_id", "split", "case_type", "pitch", "expected_outcome", "reference_show_id", "grading_note")
FREEZE_DIR = os.path.join(HERE, "freeze")
FREEZE_FILE = os.path.join(FREEZE_DIR, "frozen.json")
HOLDOUT_LOG = os.path.join(FREEZE_DIR, "holdout_runs.jsonl")
ABORT_KINDS = ("missing_key", "config", "http_401", "http_403", "http_404")  # every later case would fail the same way


# --------------------------------------------------------------------------- cases and grading

def load_cases(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows or not set(REQUIRED) <= set(rows[0]):
        raise ValueError(f"test cases need columns {list(REQUIRED)}")
    ids = [r["test_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("test_id values must be unique")
    for r in rows:
        if r["expected_outcome"] not in OUTCOME_TO_VERDICT:
            raise ValueError(f"{r['test_id']}: unknown expected_outcome {r['expected_outcome']!r}")
        if r["split"] not in ("dev", "holdout"):
            raise ValueError(f"{r['test_id']}: split must be dev or holdout")
    return rows


def grade(case, res):
    """Score what code can score. The explanation is graded by a person against grading_note."""
    exp = OUTCOME_TO_VERDICT[case["expected_outcome"]]
    status = res.get("status")
    claimed = res.get("claimed_verdict") or res.get("verdict")
    if status in ("provider_error", "internal_error", "invalid_input"):
        claimed = None
    rep = as_dict(res.get("report"))
    gid = rep.get("closest_id") if isinstance(rep.get("closest_id"), str) else None
    ref = case["reference_show_id"]
    label_ok = claimed == exp
    verified = status == "verified"
    if exp == "same_sequence":
        ref_ok = bool(label_ok and gid == ref)
    elif exp == "shared_trope_only" and ref:
        ref_ok = "same as reference" if gid == ref else ("different reference: check the explanation" if gid else "no reference given")
    else:
        ref_ok = "n/a"
    return {
        "expected_verdict": exp, "claimed_verdict": claimed, "status": status, "status_reason": res.get("status_reason"),
        "got_id": gid, "label_ok": label_ok,
        "verified_success": bool(label_ok and verified and (ref_ok is True if exp == "same_sequence" else True)),
        "reference_ok": ref_ok,
        "false_alarm": exp != "same_sequence" and claimed == "same_sequence",
    }


def counts(rows):
    n = len(rows)
    seq = [r for r in rows if r["exp"] == "same_sequence"]
    trope = [r for r in rows if r["exp"] == "shared_trope_only"]
    unrel = [r for r in rows if r["exp"] == "no_meaningful_match"]
    vague = [r for r in rows if r["exp"] == "insufficient_information"]
    qv, qc = sum(r["qv"] for r in rows), sum(r["qc"] for r in rows)
    st = lambda s: sum(1 for r in rows if r["status"] == s)  # noqa: E731
    out = {
        "cases": n,
        "label agreement (any status)": f"{sum(r['label_ok'] for r in rows)}/{n}",
        "verified success (label right AND source checks passed)": f"{sum(r['verified_success'] for r in rows)}/{n}",
        "label right but NOT verified": f"{sum(1 for r in rows if r['label_ok'] and not r['verified_success'])}/{n}",
        "unverified / unsupported reports": f"{st('unverified')}/{n}",
        "provider errors (no comparison made)": f"{st('provider_error')}/{n}",
        "tool errors / invalid input": f"{st('internal_error') + st('invalid_input')}/{n}",
        "sequence_overlap: intended show named, label right": f"{sum(1 for r in seq if r['label_ok'] and r['reference_ok'] is True)}/{len(seq)}",
        "sequence_overlap: intended show named AND verified": f"{sum(1 for r in seq if r['verified_success'])}/{len(seq)}",
        "trope_only: false alarms (claimed same_sequence)": f"{sum(r['false_alarm'] for r in trope)}/{len(trope)}",
        "trope_only: label right (any status) / verified": f"{sum(r['label_ok'] for r in trope)}/{len(trope)} / {sum(r['verified_success'] for r in trope)}/{len(trope)}",
        "unrelated: false matches (claimed a match)": f"{sum(1 for r in unrel if r['claimed'] in ('same_sequence', 'shared_trope_only'))}/{len(unrel)}",
        "unrelated: no_meaningful_match (any status) / verified": f"{sum(r['label_ok'] for r in unrel)}/{len(unrel)} / {sum(r['verified_success'] for r in unrel)}/{len(unrel)}",
        "underspecified: asked for more (any status) / verified": f"{sum(r['label_ok'] for r in vague)}/{len(vague)} / {sum(r['verified_success'] for r in vague)}/{len(vague)}",
        "quoted passages found in source text": f"{qv}/{qc}",
        "cases where a near-miss quote was corrected to the source wording": f"{sum(1 for r in rows if r.get('repaired'))}/{n}",
        "cases that needed the one feedback retry / verified only after it": f"{sum(1 for r in rows if r.get('retried'))}/{n} / {sum(1 for r in rows if r.get('retried') and r['status'] == 'verified')}/{n}",
    }
    base = [r for r in seq if r["base_rank"] is not None]
    out["KEYWORD BASELINE (retrieval only; no verdicts): intended show ranked 1st"] = f"{sum(1 for r in base if r['base_rank'] == 1)}/{len(seq)}"
    out["KEYWORD BASELINE: intended show in top 3"] = f"{sum(1 for r in base if r['base_rank'] <= 3)}/{len(seq)}"
    out["KEYWORD BASELINE: intended show in top 5"] = f"{sum(1 for r in base if r['base_rank'] <= 5)}/{len(seq)}"
    return out


# --------------------------------------------------------------------------- freeze

def sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def file_sha(path):
    with open(path, "rb") as f:
        return sha(f.read())


def code_sha():
    h = hashlib.sha256()
    for rel in ("pitchcheck/prompts.py", "pitchcheck/pipeline.py", "pitchcheck/verify.py", "pitchcheck/llm.py"):
        with open(os.path.join(HERE, rel), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def current_fingerprint(cfg, catalog_path, cases_path):
    return {
        "prompt_version": prompts.PROMPT_VERSION,
        "prompt_hash": sha((prompts.EXTRACT + prompts.COMPARE).encode("utf-8")),
        "code_hash": code_sha(),
        "config": public_config(cfg),
        "mock": is_mock(cfg),
        "catalog_hash": file_sha(catalog_path),
        "test_cases_hash": file_sha(cases_path),
        "tool_version": TOOL_VERSION,
    }


def freeze_id(fp):
    return sha(json.dumps(fp, sort_keys=True).encode("utf-8"))


def do_freeze(cfg, catalog_path, cases_path):
    fp = current_fingerprint(cfg, catalog_path, cases_path)
    os.makedirs(FREEZE_DIR, exist_ok=True)
    record = dict(fp, freeze_id=freeze_id(fp), frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(FREEZE_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"Frozen: freeze_id {record['freeze_id']}  prompt {fp['prompt_version']}/{fp['prompt_hash']}  provider {fp['config']['provider']}/{fp['config']['model']}")
    if fp["mock"]:
        print("NOTE: this freeze is for the MOCK provider. Re-freeze with your real provider after tuning, before the holdout run.")
    print(f"Saved to {FREEZE_FILE}. Any later change to the prompt, code, config, catalog or labels breaks the freeze.")
    return record


def check_freeze(cfg, catalog_path, cases_path, allow_repeat):
    if not os.path.isfile(FREEZE_FILE):
        sys.exit("No freeze found. Tune on the dev cases, then run `python eval.py --freeze`, then the holdout.")
    with open(FREEZE_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    now = current_fingerprint(cfg, catalog_path, cases_path)
    changed = [k for k in now if saved.get(k) != now[k]]
    if changed:
        sys.exit("The frozen configuration no longer matches (changed: " + ", ".join(changed) + "). "
                 "If you changed these deliberately, the holdout is no longer clean for tuning decisions: re-freeze, and write fresh holdout cases if you "
                 "changed anything after seeing holdout results.")
    if os.path.isfile(HOLDOUT_LOG) and not allow_repeat:
        with open(HOLDOUT_LOG, encoding="utf-8") as f:
            earlier = [json.loads(line) for line in f if line.strip()]
        if any(e.get("freeze_id") == saved["freeze_id"] for e in earlier):
            sys.exit("The holdout was already run for this freeze. Re-running it after changes would make it a second dev set. "
                     "Report the first holdout result, or write fresh holdout cases. (--allow-repeat overrides, and the repeat is recorded.)")
    return saved


# --------------------------------------------------------------------------- main

def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--catalog", default=os.path.join(HERE, "data", "catalog.csv"))
    ap.add_argument("--cases", default=os.path.join(HERE, "data", "test_cases.csv"))
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--freeze", action="store_true", help="record the current prompt/config/data hashes and exit")
    ap.add_argument("--allow-repeat", action="store_true", help="re-run the holdout for the same freeze (recorded as a repeat)")
    ap.add_argument("--mode", choices=["full", "shortlist"], default=None, help="full (default) compares all catalog shows")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-root", default=os.path.join(HERE, "outputs", "eval"), help="where raw outputs are written")
    a = ap.parse_args()

    cfg = load_config(a.config)
    catalog = load_catalog(a.catalog)
    cases = load_cases(a.cases)

    cat_text = " || ".join(norm(r["synopsis"]) for r in catalog)
    leaked = [c["test_id"] for c in cases if norm(c["pitch"]) in cat_text]
    if leaked:
        sys.exit(f"test pitches found inside the catalog (remove them): {leaked}")
    known = {r["id"] for r in catalog}
    bad_refs = [c["test_id"] for c in cases if c["reference_show_id"] and c["reference_show_id"] not in known]
    if bad_refs:
        sys.exit(f"reference_show_id not in the catalog for: {bad_refs}")

    if a.freeze:
        do_freeze(cfg, a.catalog, a.cases)
        return 0

    frozen = None
    if a.split in ("holdout", "all"):
        frozen = check_freeze(cfg, a.catalog, a.cases, a.allow_repeat)

    run_cases = [c for c in cases if a.split == "all" or c["split"] == a.split]
    fp = current_fingerprint(cfg, a.catalog, a.cases)
    label = a.label or f"{cfg['provider']}-{cfg['model']}-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}".replace("/", "_")
    out = os.path.join(a.out_root, label)
    os.makedirs(os.path.join(out, "cases"), exist_ok=True)
    cache = CardCache(os.path.join(out, "cards_cache.json"))
    docs = {r["id"]: r["synopsis"] for r in catalog}

    rows, aborted = [], None
    for c in run_cases:
        res = check_pitch(c["pitch"], catalog, cfg, cache, mode=a.mode)  # the app sees the pitch only
        g = grade(c, res)
        order = [i for i, _ in rank(c["pitch"], docs)]
        base_rank = order.index(c["reference_show_id"]) + 1 if c["reference_show_id"] in order else None
        res["case"], res["grade"], res["keyword_baseline_rank_of_reference"] = c, g, base_rank
        res["fingerprint"] = fp
        with open(os.path.join(out, "cases", c["test_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        chk = as_dict(res.get("checks"))
        rows.append({
            "test_id": c["test_id"], "split": c["split"], "case_type": c["case_type"], "exp": g["expected_verdict"],
            "claimed": g["claimed_verdict"], "got_id": g["got_id"], "reference": c["reference_show_id"], "status": g["status"],
            "status_reason": g["status_reason"], "label_ok": g["label_ok"], "verified_success": g["verified_success"],
            "reference_ok": g["reference_ok"], "false_alarm": g["false_alarm"],
            "repaired": len(as_list(res.get("quote_repairs"))), "retried": bool(res.get("retry")), "qv": chk.get("quotes_verified", 0), "qc": chk.get("quotes_checked", 0), "base_rank": base_rank, "grading_note": c["grading_note"],
        })
        mark = "VERIFIED-OK" if g["verified_success"] else ("label-only " if g["label_ok"] else "MISS       ")
        print(f"{mark} {c['test_id']} [{c['split']}] expected={g['expected_verdict']} claimed={g['claimed_verdict']} id={g['got_id']} status={g['status']}/{g['status_reason']}")
        pe = as_dict(res.get("provider_error"))
        if res.get("status") == "provider_error" and pe.get("kind") in ABORT_KINDS:
            aborted = pe
            print(f"\nPROVIDER ERROR ({pe.get('kind')}): {pe.get('message')}\nWhat to do: {pe.get('hint')}\nStopping: every remaining case would fail the same way.")
            break

    if aborted:
        return 3

    with open(os.path.join(out, "grading_sheet.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_id", "split", "case_type", "expected_verdict", "claimed_verdict", "status", "status_reason", "label_ok", "verified_success",
                    "reference_show_id", "got_id", "reference_ok", "quotes_found", "keyword_baseline_rank", "grading_note",
                    "explanation_grade (you fill: pass/partial/fail)", "unsupported_claims_found (you fill)", "your_notes"])
        for r in rows:
            w.writerow([r["test_id"], r["split"], r["case_type"], r["exp"], r["claimed"], r["status"], r["status_reason"], r["label_ok"], r["verified_success"],
                        r["reference"], r["got_id"], r["reference_ok"], f"{r['qv']}/{r['qc']}", r["base_rank"], r["grading_note"], "", "", ""])

    summary = {"label": label, "provider": cfg["provider"], "model": cfg["model"], "mock": is_mock(cfg),
               "mode": a.mode or cfg.get("retrieval", {}).get("mode", "full"), "split_run": a.split, "fingerprint": fp,
               "freeze_id": frozen["freeze_id"] if frozen else None}
    for name in ("dev", "holdout"):
        sub = [r for r in rows if r["split"] == name]
        if sub:
            summary[name] = counts(sub)
    summary["rows"] = [{k: v for k, v in r.items() if k != "grading_note"} for r in rows]
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    md = ["# Evaluation summary", ""]
    if is_mock(cfg):
        md += ["> **MOCK PROVIDER.** A plumbing check only. These are not model results and must not be reported as accuracy.", ""]
    md += [f"- provider / model: {cfg['provider']} / {cfg['model']}", f"- comparison: {summary['mode']} (all {len(catalog)} shows unless 'shortlist')",
           f"- prompt: {fp['prompt_version']} / {fp['prompt_hash']}   code: {fp['code_hash']}",
           f"- freeze id: {summary['freeze_id'] or 'not frozen (dev run)'}", f"- split run: {a.split}", ""]
    for name in ("dev", "holdout"):
        if name in summary:
            md += [f"## {name} cases", ""] + [f"- {k}: {v}" for k, v in summary[name].items()] + [""]
    md += ["Explanations (shared events, the divergence, evidence from both texts) are graded by you in grading_sheet.csv.",
           "With 12 authored cases these counts show behaviours and failures; they are not an estimate of production accuracy."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nraw outputs: {out}")

    if frozen is not None:
        os.makedirs(FREEZE_DIR, exist_ok=True)
        with open(HOLDOUT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"freeze_id": frozen["freeze_id"], "label": label, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "repeat": bool(a.allow_repeat), "mock": is_mock(cfg), "split": a.split}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
