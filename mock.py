"""Offline stand-ins for the model. They exist ONLY to test the plumbing (parsing, caching,
verification, status handling, UI, evaluation harness). Their outputs are not model results
and must never be reported as accuracy.

mock         - deterministic, crude lexical matcher
mock_faulty  - deliberately fabricates titles and quotes to prove the checks catch them
"""
import json

from .textutil import sentences, tokens

KIND_OF_ROLE = {"opening_conflict": "opening_conflict", "protagonist_choice": "decision", "reveal": "reveal", "other": "consequence"}


def _jacc(a, b):
    a, b = set(tokens(a)), set(tokens(b))
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _extract(text):
    ss = sentences(text)[:6]
    events = []
    for i, s in enumerate(ss):
        if i == 0:
            role = "opening_conflict"
        elif i == len(ss) - 1 and len(ss) >= 3:
            role = "reveal"
        elif i == 1:
            role = "protagonist_choice"
        else:
            role = "other"
        events.append({"role": role, "text": s, "excerpt": s})
    return {"events": events}


def _compare(ctx):
    pitch_events = ctx["pitch"]["events"]
    best = None
    for cat in ctx["catalog"]:
        used, last, matches, total = set(), -1, [], 0.0
        for pe in pitch_events:
            cand = None
            for j, ce in enumerate(cat["events"]):
                if j <= last or j in used:
                    continue
                sim = _jacc(pe["text"], ce["text"])
                if sim >= 0.2 and (cand is None or sim > cand[1]):
                    cand = (j, sim, ce)
            if cand:
                used.add(cand[0])
                last = cand[0]
                total += cand[1]
                matches.append((pe, cand[2]))
        key = (len(matches), total)
        if best is None or key > best[0]:
            best = (key, cat, matches)
    (n, _), cat, matches = best
    verdict = "same_sequence" if n >= 2 else ("shared_trope_only" if n == 1 else "no_meaningful_match")
    if n == 0:
        return {"verdict": verdict, "closest_id": None, "shared_events": [], "order_basis": "unclear", "first_difference": None,
                "editor_question": "Mock output: is there a decision in the pitch the catalog never reaches?", "missing_info": []}
    shared = [
        {"pitch_excerpt": pe["excerpt"], "catalog_excerpt": ce["excerpt"], "what_is_shared": "similar wording (mock)",
         "kind": KIND_OF_ROLE.get(pe["role"], "consequence"), "pitch_agency": "protagonist_chooses", "catalog_agency": "protagonist_chooses"}
        for pe, ce in matches
    ]
    matched = {pe["text"] for pe, _ in matches}
    diff = next((e for e in pitch_events if e["text"] not in matched), None)
    return {
        "verdict": verdict,
        "closest_id": cat["id"],
        "shared_events": shared,
        "order_basis": "unclear",
        "first_difference": {"established": False, "explanation": "Mock output: the difference was not assessed.",
                             "pitch_excerpt": diff["excerpt"] if diff else None, "catalog_excerpt": None},
        "editor_question": "Mock output: does the unmatched event change what the viewer is promised?",
        "missing_info": [],
    }


def _faulty(task, ctx):
    if task == "extract":
        return {"events": [
            {"role": "opening_conflict", "text": "invented", "excerpt": "an invented sentence that is not in the source"},
            {"role": "protagonist_choice", "text": "invented", "excerpt": "another fabricated quotation about a secret"},
        ]}
    return {
        "verdict": "same_sequence",
        "closest_id": "SHOW_999",
        "shared_events": [
            {"pitch_excerpt": "this quote never appears in the pitch", "catalog_excerpt": "nor does this one appear in the show",
             "what_is_shared": "fabricated", "kind": "decision", "pitch_agency": "protagonist_chooses", "catalog_agency": "protagonist_chooses"},
            {"pitch_excerpt": "second fabricated pitch quotation", "catalog_excerpt": "second fabricated catalog quotation",
             "what_is_shared": "fabricated", "kind": "reveal", "pitch_agency": "discovery", "catalog_agency": "discovery"},
        ],
        "order_basis": "stated",
        "first_difference": None,
        "editor_question": "n/a",
        "missing_info": [],
    }


def mock_complete(provider, task, context):
    if provider == "mock_faulty":
        return json.dumps(_faulty(task, context))
    if task == "extract":
        return json.dumps(_extract(context["text"]))
    if task == "compare":
        return json.dumps(_compare(context))
    raise ValueError(f"unknown mock task {task!r}")
