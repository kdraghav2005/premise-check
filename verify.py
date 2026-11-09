"""Mechanical checks that do NOT need a model.

What Python can certify: the show id exists; each quoted passage occurs in the pitch or in the selected
synopsis; where it occurs; that fields have the right types; that a verdict is consistent with its fields.
What Python cannot certify: story chronology, motive, or that the model's interpretation is right.
A claim that fails a check is shown as unverified, never silently repaired or hidden."""
from .textutil import find_range, word_count

ROLES = ("opening_conflict", "protagonist_choice", "reveal", "other")
KEY_ROLES = ("opening_conflict", "protagonist_choice", "reveal")
VERDICTS = ("same_sequence", "shared_trope_only", "no_meaningful_match", "insufficient_information")
MATCH_VERDICTS = ("same_sequence", "shared_trope_only")
KINDS = ("opening_conflict", "decision", "reveal", "consequence")
AGENCY = ("protagonist_chooses", "protagonist_forced", "other_acts", "discovery")
ORDER_BASIS = ("stated", "unclear")
MIN_EXCERPT_WORDS = 3
MAX_EXCERPT_WORDS = 40

SCOPE_NOTE = (
    "Mechanical checks only. Python confirms that the show id exists, that each quoted passage occurs in the pitch or in the "
    "selected synopsis, and where it occurs. It does not certify story chronology, motive, or that the interpretation is "
    "correct. Order checks compare positions in the text, and a blurb's sentence order may not be the story's order."
)

ABSOLUTE_WORDS = (
    " never ", " does not happen", " doesn't happen", " no such ", " there is no ", " is absent", " does not occur",
    " does not include", " doesn't include", " does not have", " doesn't have", " without any ",
)


def _is_str(x):
    return isinstance(x, str) and x.strip() != ""


def _in_bounds(excerpt):
    n = word_count(excerpt)
    return MIN_EXCERPT_WORDS <= n <= MAX_EXCERPT_WORDS


def _locate(excerpt, text):
    """(start, end) if excerpt is a plausible-length passage found in text, else None."""
    if not isinstance(excerpt, str) or not _in_bounds(excerpt):
        return None
    return find_range(excerpt, text)


# --------------------------------------------------------------------------- extraction

def validate_extraction(data, source_text):
    """Keep only events whose excerpt really occurs in source_text.
    well_formed=False means the reply was not an object with an events list (a model failure, not a vague pitch)."""
    out = {"well_formed": False, "events": [], "dropped": [], "order_ok": True, "missing_roles": list(KEY_ROLES)}
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        out["dropped"].append({"reason": "the reply was not an object with an 'events' list"})
        return out
    out["well_formed"] = True
    kept, positions = [], []
    for ev in data["events"]:
        if not isinstance(ev, dict):
            out["dropped"].append({"reason": "event is not an object", "event": str(ev)[:80]})
            continue
        text, excerpt = ev.get("text"), ev.get("excerpt")
        if not _is_str(text) or not isinstance(excerpt, str):
            out["dropped"].append({"reason": "event 'text' or 'excerpt' missing or not text", "event": str(ev)[:80]})
            continue
        rng = _locate(excerpt, source_text)
        if rng is None:
            out["dropped"].append({"reason": "excerpt not found in the source, or outside 3-40 words", "excerpt": excerpt[:120]})
            continue
        if any(rng[0] < k["end"] and k["pos"] < rng[1] for k in kept):
            out["dropped"].append({"reason": "excerpt overlaps an earlier event (not a distinct event)", "excerpt": excerpt[:120]})
            continue
        role = ev.get("role") if ev.get("role") in ROLES else "other"
        kept.append({"role": role, "text": text.strip(), "excerpt": excerpt.strip(), "pos": rng[0], "end": rng[1]})
        positions.append(rng[0])
    out["events"] = kept
    out["order_ok"] = all(a <= b for a, b in zip(positions, positions[1:]))
    present = {e["role"] for e in kept}
    out["missing_roles"] = [r for r in KEY_ROLES if r not in present]
    return out


# --------------------------------------------------------------------------- comparison report

def check_report(report, pitch_text, catalog_by_id, pitch_stats=None):
    """Verify a comparison report. Never raises on wrong JSON types.
    pitch_stats: {"events": verified pitch events, "words": pitch word count}."""
    items, failures, warnings = [], [], []
    q = {"checked": 0, "verified": 0}

    def add(key, label, ok, detail=""):
        items.append({"id": key, "label": label, "passed": bool(ok), "detail": detail})
        if not ok:
            failures.append(label + (f" ({detail})" if detail else ""))
        return bool(ok)

    def finish(counted=0, mismatches=None):
        return {
            "ok": not failures,
            "items": items,
            "failures": failures,
            "warnings": warnings,
            "quotes_checked": q["checked"],
            "quotes_verified": q["verified"],
            "counted_events": counted,
            "agency_mismatches": mismatches or [],
            "scope_note": SCOPE_NOTE,
        }

    if not isinstance(report, dict):
        add("shape", "the report is a JSON object", False, type(report).__name__)
        return finish()

    verdict = report.get("verdict")
    if not add("verdict", "verdict is one of the four editorial verdicts", verdict in VERDICTS if isinstance(verdict, str) else False, repr(verdict)[:60]):
        return finish()

    # ---- field types (wrong types fail the check, they never crash)
    cid = report.get("closest_id")
    cid_ok = cid is None or isinstance(cid, str)
    add("types_id", "closest_id is text or empty", cid_ok, type(cid).__name__)
    cid = cid.strip() if isinstance(cid, str) and cid.strip() else None

    shared = report.get("shared_events")
    shared_ok = shared is None or isinstance(shared, list)
    add("types_shared", "shared_events is a list", shared_ok, type(shared).__name__)
    shared = shared if isinstance(shared, list) else []

    fd = report.get("first_difference")
    fd_ok = fd is None or isinstance(fd, dict)
    add("types_fd", "first_difference is an object or empty", fd_ok, type(fd).__name__)
    fd = fd if isinstance(fd, dict) else None

    question = report.get("editor_question")
    add("types_question", "editor_question is text", question is None or isinstance(question, str), type(question).__name__)
    question = question.strip() if isinstance(question, str) else ""

    mi = report.get("missing_info")
    add("types_missing", "missing_info is a list", mi is None or isinstance(mi, list), type(mi).__name__)

    # ---- verdicts that must carry no match
    if verdict not in MATCH_VERDICTS:
        problems = []
        if cid:
            problems.append("closest_id is set")
        if shared:
            problems.append("shared_events is not empty")
        if fd:
            problems.append("first_difference is set")
        add("consistency", f"{verdict} carries no matched show, shared events or first difference", not problems, "; ".join(problems))
        stats = pitch_stats or {}
        if verdict == "no_meaningful_match":
            if "events" in stats:
                add("no_match_scope", "a no-match finding needs a pitch with at least two stated events", stats["events"] >= 2, f"{stats['events']} event(s) found")
        else:
            add("follow_up", "insufficient_information asks one specific question", bool(question) and question.count("?") == 1, question[:80] or "no question")
            if "events" in stats and "words" in stats:
                plausible = not (stats["events"] >= 3 and stats["words"] >= 30)
                add("insufficient_plausible", "the pitch is thin enough for insufficient_information", plausible, f"{stats['events']} events in {stats['words']} words")
        return finish()

    # ---- match verdicts
    ctext = None
    if cid is None:
        add("show_id", "the verdict names a closest show", False, "closest_id is empty")
    elif cid not in catalog_by_id:
        add("show_id", "closest_id exists in the catalog", False, repr(cid)[:60])
    else:
        add("show_id", "closest_id exists in the catalog", True, cid)
        ctext = catalog_by_id[cid]["synopsis"]

    events = []
    for i, ev in enumerate(shared, 1):
        if not isinstance(ev, dict):
            add(f"ev{i}_shape", f"shared event {i} is an object", False, type(ev).__name__)
            continue
        pe, ce = ev.get("pitch_excerpt"), ev.get("catalog_excerpt")
        if not isinstance(pe, str) or not isinstance(ce, str):
            add(f"ev{i}_text", f"shared event {i} has text excerpts for both sides", False)
            continue
        q["checked"] += 2
        pr = _locate(pe, pitch_text)
        cr = _locate(ce, ctext) if ctext is not None else None
        if pr:
            q["verified"] += 1
        if cr:
            q["verified"] += 1
        add(f"ev{i}_pitch", f"shared event {i}: quoted pitch text occurs in the pitch", bool(pr), "" if pr else pe[:70])
        add(f"ev{i}_catalog", f"shared event {i}: quoted synopsis text occurs in the selected synopsis", bool(cr), "" if cr else ce[:70])
        kind, pa, ca = ev.get("kind"), ev.get("pitch_agency"), ev.get("catalog_agency")
        enums_ok = kind in KINDS and pa in AGENCY and ca in AGENCY if all(isinstance(x, str) for x in (kind, pa, ca)) else False
        add(f"ev{i}_labels", f"shared event {i}: kind and who-acts labels are valid", enums_ok, f"{kind!r}, {pa!r}, {ca!r}" if not enums_ok else "")
        add(f"ev{i}_what", f"shared event {i}: says who acts and why it matters", _is_str(ev.get("what_is_shared")))
        events.append({"i": i, "pr": pr, "cr": cr, "ok": bool(pr and cr and enums_ok and _is_str(ev.get("what_is_shared"))), "kind": kind, "pa": pa, "ca": ca})

    good = [e for e in events if e["ok"]]

    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    distinct = True
    for x in range(len(good)):
        for y in range(x + 1, len(good)):
            if overlap(good[x]["pr"], good[y]["pr"]) or overlap(good[x]["cr"], good[y]["cr"]):
                distinct = False
    add("distinct", "shared events are distinct (their quoted passages do not overlap)", distinct)

    in_order = all(a["pr"][0] < b["pr"][0] and a["cr"][0] < b["cr"][0] for a, b in zip(good, good[1:]))
    add("text_order", "quoted passages follow the same order in the pitch and in the synopsis (text order only)", in_order)

    counted = [e for e in good if e["pa"] == e["ca"]]
    mismatches = [e["i"] for e in good if e["pa"] != e["ca"]]
    if mismatches:
        warnings.append(
            "Who acts differs in shared event(s) " + ", ".join(map(str, mismatches))
            + " (for example a choice versus something forced); they are not counted toward a sequence match."
        )

    if verdict == "same_sequence":
        kinds = {e["kind"] for e in counted}
        add("sequence_rule", "a sequence match has at least two distinct events of at least two kinds, with the same party acting in both texts",
            len(counted) >= 2 and len(kinds) >= 2, f"{len(counted)} counted event(s), kinds: {sorted(kinds) or 'none'}")
    else:
        add("trope_rule", "a trope match has at least one verified shared event", len(good) >= 1, f"{len(good)} verified")

    ob = report.get("order_basis")
    add("order_basis", "the report says whether the texts establish the order (stated or unclear)", ob in ORDER_BASIS if isinstance(ob, str) else False, repr(ob)[:30])
    if ob == "unclear":
        warnings.append("The texts do not clearly establish the order of events; the editor should read the paired passages before relying on the sequence.")

    # ---- first difference
    if fd is None:
        add("fd_present", "a first-difference note is given (it may say no difference is established)", False)
    else:
        est = fd.get("established")
        expl = fd.get("explanation")
        add("fd_est", "first_difference says whether a difference is established (true or false)", isinstance(est, bool), repr(est)[:20])
        add("fd_expl", "first_difference has an explanation", _is_str(expl))
        pe, ce = fd.get("pitch_excerpt"), fd.get("catalog_excerpt")
        pr = cr = None
        if est is True:
            for label, ex, text, store in (("pitch", pe, pitch_text, "pr"), ("synopsis", ce, ctext, "cr")):
                q["checked"] += 1
                r = _locate(ex, text) if text is not None else None
                if r:
                    q["verified"] += 1
                add(f"fd_{label}", f"claimed difference: supporting text from the {label} is quoted and found", bool(r), "" if r else str(ex)[:70])
                if store == "pr":
                    pr = r
                else:
                    cr = r
            if pr and cr and any(e["pr"] == pr and e["cr"] == cr for e in good):
                add("fd_new", "the claimed difference does not just repeat a shared event", False)
            if _is_str(expl):
                low = " " + expl.lower() + " "
                if any(w in low for w in ABSOLUTE_WORDS):
                    warnings.append("The difference note uses absolute wording about the existing show. A short synopsis may omit details; read it as 'the synopsis says X', not 'the show never does Y'.")
        elif est is False:
            warnings.append("No first difference is established by the two texts. A detail missing from a short blurb does not show the existing show lacks it.")
            for label, ex, text in (("pitch", pe, pitch_text), ("synopsis", ce, ctext)):
                if ex is not None:
                    q["checked"] += 1
                    r = _locate(ex, text) if (isinstance(ex, str) and text is not None) else None
                    if r:
                        q["verified"] += 1
                    add(f"fd_{label}", f"the {label} passage quoted in the difference note is found", bool(r), "" if r else str(ex)[:70])

    return finish(len(counted), mismatches)
