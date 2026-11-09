#!/usr/bin/env python3
"""A deliberately naive baseline for comparison with Premise Check: ONE prompt, no event extraction, no verification, no retry.

  python naive_baseline.py             run all 12 test cases once (needs a real provider in config.json and the key in the environment)

It asks the same model the same question in the simplest way ("is this a repeat? quote your evidence") and then applies the SAME
source check afterwards, only to measure it: are the quoted passages really in the pitch / the chosen synopsis?
This answers one question: how often does a plain prompt give a right-looking verdict backed by quotes that are not in the text,
which the verifier in Premise Check would have rejected. It is not a tuned competitor: the prompt below was written once and not changed.
It does not touch the freeze record or the holdout log, and it is not part of the app."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval import OUTCOME_TO_VERDICT, load_cases  # noqa: E402
from pitchcheck.llm import ModelOutputError, ProviderError, complete, is_mock, load_config, parse_json, public_config  # noqa: E402
from pitchcheck.pipeline import load_catalog  # noqa: E402
from pitchcheck.safe import as_dict, as_list  # noqa: E402
from pitchcheck.textutil import find_range, setup_stdout  # noqa: E402

VERDICTS = ("same_sequence", "shared_trope_only", "no_meaningful_match", "insufficient_information")
ABORT_KINDS = ("missing_key", "config", "http_401", "http_403", "http_404")

NAIVE_PROMPT = """You help a story editor. Decide whether the NEW PITCH repeats the sequence of events of one of the CATALOG shows.

Verdicts:
- same_sequence: the same key events happen in the same order, with the same party acting.
- shared_trope_only: they share a device (for example a staged marriage) but the story progresses differently.
- no_meaningful_match: nothing in the catalog resembles the pitch.
- insufficient_information: the pitch does not say enough to judge.

Reply with JSON only:
{"verdict":"same_sequence|shared_trope_only|no_meaningful_match|insufficient_information","closest_id":"<catalog id or null>","evidence":[{"pitch_quote":"...","synopsis_quote":"..."}],"reason":"..."}
Quote the evidence word for word from the texts.

NEW PITCH:
%s

CATALOG:
%s
"""


def build_prompt(pitch, catalog):
    cat = "\n\n".join(f"[id: {r['id']}] {r['title']}\n{r['synopsis']}" for r in catalog)
    return NAIVE_PROMPT % (pitch, cat)


def grade_reply(raw, case, by_id):
    """Grade one raw reply. Pure function, no model calls."""
    exp = OUTCOME_TO_VERDICT[case["expected_outcome"]]
    g = {"expected": exp, "parsed": False, "verdict": None, "closest_id": None, "label_ok": False, "intended_show_named": None,
         "quotes_checked": 0, "quotes_found": 0, "quotes_missing": [], "match_without_evidence": False}
    try:
        rep = parse_json(raw)
    except ModelOutputError:
        return g
    if not isinstance(rep, dict):
        return g
    g["parsed"] = True
    verdict = rep.get("verdict") if rep.get("verdict") in VERDICTS else None
    cid = rep.get("closest_id") if isinstance(rep.get("closest_id"), str) and rep.get("closest_id") in by_id else None
    g["verdict"], g["closest_id"] = verdict, cid
    g["label_ok"] = verdict == exp
    if exp == "same_sequence":
        g["intended_show_named"] = bool(cid and cid == case["reference_show_id"])
    ev = [as_dict(e) for e in as_list(rep.get("evidence"))]
    ctext = by_id[cid]["synopsis"] if cid else None
    for e in ev:
        for key, text in (("pitch_quote", case["pitch"]), ("synopsis_quote", ctext)):
            q = e.get(key)
            if not isinstance(q, str) or not q.strip():
                continue
            g["quotes_checked"] += 1
            if text is not None and find_range(q, text):
                g["quotes_found"] += 1
            else:
                g["quotes_missing"].append(q[:100])
    if verdict in ("same_sequence", "shared_trope_only") and not ev:
        g["match_without_evidence"] = True
    return g


def counts(rows):
    n = len(rows)
    ok_label = [r for r in rows if r["label_ok"]]
    clean = [r for r in rows if r["parsed"] and not r["quotes_missing"] and not r["match_without_evidence"]]
    return {
        "cases": n,
        "reply not valid JSON": f"{sum(1 for r in rows if not r['parsed'])}/{n}",
        "provider errors": f"{sum(1 for r in rows if r.get('provider_error'))}/{n}",
        "label agreement": f"{len(ok_label)}/{n}",
        "label right AND every quote found in the source text (what the verifier would accept)": f"{sum(1 for r in ok_label if r in clean)}/{n}",
        "label right but at least one quote NOT in the source text": f"{sum(1 for r in ok_label if r['quotes_missing'])}/{n}",
        "cases with at least one quote not in the source text (any label)": f"{sum(1 for r in rows if r['quotes_missing'])}/{n}",
        "quoted passages found in source text": f"{sum(r['quotes_found'] for r in rows)}/{sum(r['quotes_checked'] for r in rows)}",
        "match verdict given with no evidence quoted": f"{sum(1 for r in rows if r['match_without_evidence'])}/{n}",
    }


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--catalog", default=os.path.join(HERE, "data", "catalog.csv"))
    ap.add_argument("--cases", default=os.path.join(HERE, "data", "test_cases.csv"))
    ap.add_argument("--out-root", default=os.path.join(HERE, "outputs", "eval"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    if is_mock(cfg):
        sys.exit("The naive baseline needs a real provider (set config.json to Groq and put the key in PITCHCHECK_API_KEY). A mock run would mean nothing.")
    catalog = load_catalog(a.catalog)
    by_id = {r["id"]: r for r in catalog}
    cases = load_cases(a.cases)
    label = f"{cfg['provider']}-{cfg['model']}-NAIVE-all-{time.strftime('%Y%m%d-%H%M%S')}".replace("/", "_")
    out = os.path.join(a.out_root, label)
    os.makedirs(os.path.join(out, "cases"), exist_ok=True)

    rows = []
    for c in cases:
        prompt = build_prompt(c["pitch"], catalog)
        rec = {"test_id": c["test_id"], "split": c["split"], "pitch": c["pitch"], "prompt": prompt, "raw_reply": None, "provider_error": None}
        try:
            rec["raw_reply"] = complete(prompt, cfg, task="naive", context=None)
        except ProviderError as e:
            rec["provider_error"] = e.to_dict()
        g = grade_reply(rec["raw_reply"], c, by_id) if rec["raw_reply"] is not None else grade_reply("", c, by_id)
        rec["grade"] = g
        with open(os.path.join(out, "cases", c["test_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        row = dict(g, test_id=c["test_id"], split=c["split"], provider_error=bool(rec["provider_error"]))
        rows.append(row)
        status = "PROVIDER-ERROR" if rec["provider_error"] else ("label-ok " if g["label_ok"] else "label-miss")
        print(f"{status} {c['test_id']} [{c['split']}] expected={g['expected']} said={g['verdict']} id={g['closest_id']} quotes {g['quotes_found']}/{g['quotes_checked']}")
        if rec["provider_error"] and rec["provider_error"].get("kind") in ABORT_KINDS:
            print("Stopping: every later case would fail the same way.")
            break

    summary = {"label": label, "baseline": "naive single prompt (no extraction, no verification, no retry)", "config": public_config(cfg),
               "all": counts(rows)}
    for name in ("dev", "holdout"):
        sub = [r for r in rows if r["split"] == name]
        if sub:
            summary[name] = counts(sub)
    summary["rows"] = rows
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    md = ["# Naive baseline summary", "", "One plain prompt, same model, same 12 cases, one run each. No event extraction, no verification, no retry.",
          f"- provider / model: {cfg['provider']} / {cfg['model']}", ""]
    for name in ("all", "dev", "holdout"):
        if name in summary:
            md += [f"## {name}", ""] + [f"- {k}: {v}" for k, v in summary[name].items()] + [""]
    md += ["The quote check after the fact uses the same text-matching rule as Premise Check's verifier. 12 authored cases show behaviour, not accuracy."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nraw outputs: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
