"""The flow for one pitch, with four kinds of outcome kept visibly separate:

  verdict   (editorial)  same_sequence | shared_trope_only | no_meaningful_match | insufficient_information
  status    (technical)  verified | unverified | provider_error | invalid_input | internal_error

  verified        the model's report passed the mechanical checks (or the code gate decided a thin pitch)
  unverified      the model replied but the reply was malformed or unsupported; the raw reply is kept
  provider_error  the provider could not be used (no key, timeout, HTTP error); no comparison was made
A verdict is attached only to a verified result. An unverified result carries `claimed_verdict` for inspection."""
import csv
import hashlib
import json
import os
import time
import traceback

from . import prompts
from .llm import ModelOutputError, ProviderError, complete, is_mock, parse_json, public_config
from .runs import new_run_id
from .safe import as_dict, as_list, as_str
from .shortlist import rank
from .textutil import closest_span, find_range, word_count
from .verify import MATCH_VERDICTS, VERDICTS, check_report, validate_extraction

TOOL_VERSION = "2.1"
DETAILED_WORDS = 25  # a pitch this long cannot be called "too thin" by the code gate
MAX_PITCH_CHARS = 4000


def load_catalog(path):
    """One row per existing show: show_id (or id), title, synopsis, source_url.
    The editor's new pitch and the test pitches never belong in this file."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if "id" not in r and "show_id" in r:
            r["id"] = r["show_id"]
        rows.append(r)
    need = {"id", "title", "synopsis"}
    rows = [r for r in rows if r.get("id") or r.get("title")]
    if not rows or not need <= set(rows[0]):
        raise ValueError("catalog must have columns show_id (or id), title, synopsis (+ source_url)")
    for r in rows:
        if not r["id"] or not r["synopsis"]:
            raise ValueError(f"catalog row {r.get('title')!r} needs an id and a synopsis")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("catalog ids must be unique")
    return rows


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


class CardCache:
    """Event cards for catalog shows, cached on disk. Only well-formed, non-empty cards are cached,
    so one bad model reply can never become permanent."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except ValueError:
                self.data = {}

    def _key(self, text, cfg):
        raw = f"{cfg.get('provider')}|{cfg.get('model')}|{prompts.PROMPT_VERSION}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text, cfg, kind):
        k = self._key(text, cfg)
        if k in self.data:
            return self.data[k]
        card = extract_card(text, cfg, kind)
        if card["well_formed"] and card["events"]:
            self.data[k] = card
            if self.path:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
        return card


def extract_card(text, cfg, kind):
    """Ask the model for events and validate them. Raises ProviderError only; a bad reply is returned as data."""
    raw = complete(prompts.EXTRACT + text, cfg, task="extract", context={"text": text, "kind": kind})
    try:
        data = parse_json(raw)
        parse_error = None
    except ModelOutputError as e:
        data, parse_error = None, str(e)
    card = validate_extraction(data, text)
    card["raw"] = raw
    card["parse_error"] = parse_error
    return card


# The extraction step and the comparison step use different role words. Show the comparison step only its own vocabulary,
# otherwise it copies "other" into `kind` (seen in a live run, case T11) and the report fails its label check.
_ROLE_TO_KIND = {"opening_conflict": "opening_conflict", "protagonist_choice": "decision", "reveal": "reveal", "other": "consequence"}


def _events_for_prompt(card):
    # Only the verbatim quote goes to the comparison step. A paraphrase here gets copied back as if it were a quote
    # (seen in a live run: 6 of 6 'excerpts' were the paraphrases and failed the source check).
    return [{"kind": _ROLE_TO_KIND.get(e["role"], "consequence"), "quote": e["excerpt"]} for e in as_list(card.get("events")) if e.get("excerpt")]


def repair_quotes(report, pitch, by_id):
    """Replace a quote that is one or two words off with the exact source wording, and say so.
    Returns (report_copy, repairs). Quotes with no close match (an invented sentence) are left alone and still fail the checks."""
    if not isinstance(report, dict):
        return report, []
    rep = json.loads(json.dumps(report))
    cid = rep.get("closest_id")
    ctext = by_id[cid]["synopsis"] if isinstance(cid, str) and cid in by_id else None
    repairs = []

    def fix(holder, key, text, where):
        ex = holder.get(key)
        if not isinstance(ex, str) or not ex.strip() or text is None or find_range(ex, text):
            return
        fixed = closest_span(ex, text)
        if fixed and find_range(fixed, text):
            holder[key] = fixed
            repairs.append({"where": where, "model_quote": ex, "source_text": fixed})

    for i, ev in enumerate(as_list(rep.get("shared_events")), 1):
        if isinstance(ev, dict):
            fix(ev, "pitch_excerpt", pitch, f"shared event {i}, pitch")
            fix(ev, "catalog_excerpt", ctext, f"shared event {i}, synopsis")
    fd = rep.get("first_difference")
    if isinstance(fd, dict) and fd.get("established") is True:
        fix(fd, "pitch_excerpt", pitch, "first difference, pitch")
        fix(fd, "catalog_excerpt", ctext, "first difference, synopsis")
    return rep, repairs


def _verify(report, pitch, candidates, stats):
    """Check a report, correcting near-miss quotes first. Returns (report, checks, repairs)."""
    by_id = {c["id"]: c for c in candidates}
    fixed, repairs = repair_quotes(report, pitch, by_id)
    checks = check_report(fixed, pitch, by_id, stats)
    for r in repairs:
        checks["warnings"].append(f"Quote corrected to the source wording ({r['where']}): the model wrote \"{r['model_quote']}\", the text says \"{r['source_text']}\".")
    return fixed, checks, repairs


def build_compare_prompt(pitch, pitch_card, candidates, cards):
    parts = [prompts.COMPARE, "PITCH:\n" + pitch, "PITCH EVENTS:\n" + json.dumps(_events_for_prompt(pitch_card), ensure_ascii=False), "CATALOG:"]
    for row in candidates:
        parts.append(
            f"[id: {row['id']}] {row['title']}\nSYNOPSIS: {row['synopsis']}\n"
            f"EVENTS: {json.dumps(_events_for_prompt(cards[row['id']]), ensure_ascii=False)}"
        )
    return "\n\n".join(parts)


def assess_pitch(pitch, card):
    """Decide, from the extraction alone, whether the pitch can be compared.
    Returns (state, reason): 'ok' | 'vague' | 'unusable'.
    'vague' is reached only when the model's extraction was valid and complete (nothing discarded) and the pitch is short.
    A broken or partly discarded extraction is 'unusable': that is a model failure, never evidence of a vague pitch."""
    words = word_count(pitch)
    n = len(card["events"])
    if card.get("parse_error") or not card["well_formed"]:
        return "unusable", "the model's event list was malformed or not JSON"
    if n >= 2:
        return "ok", ""
    if card["dropped"]:
        return "unusable", f"{len(card['dropped'])} extracted event(s) were discarded (quote not found in the pitch, or malformed)"
    if words >= DETAILED_WORDS:
        return "unusable", f"the pitch has {words} words but the model listed only {n} event(s)"
    return "vague", f"the pitch states {n} event(s) in {words} words"


def follow_up_question(card):
    """One specific question about the first missing piece of story."""
    missing = card["missing_roles"]
    if "protagonist_choice" in missing:
        return "What does the protagonist decide to do in response, and what triggers that decision?"
    if "reveal" in missing:
        return "What happens as a result of that decision, or what does she discover next?"
    return "What happens at the start that sets the story in motion?"


def _new_result(pitch, cfg, mode):
    return {
        "run_id": new_run_id(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool_version": TOOL_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "pitch": pitch,
        "provider": cfg.get("provider"),
        "model": cfg.get("model"),
        "mock": is_mock(cfg),
        "config": public_config(cfg),
        "mode": mode,
        "status": "internal_error",
        "status_reason": None,
        "verdict": None,
        "claimed_verdict": None,
        "report": None,
        "checks": None,
        "provider_error": None,
        "error": None,
        "decided_by": None,
        "candidates": [],
        "baseline_top": [],
        "pitch_extraction": None,
        "catalog_warnings": [],
        "raw": {"pitch_extraction_reply": None, "compare_prompt": None, "compare_reply": None},
    }


def check_pitch(pitch, catalog, cfg, cache, mode=None, k=None):
    """Run the full flow for one pitch. Never raises: every outcome is a result dict that can be shown and saved."""
    t0 = time.time()
    rcfg = cfg.get("retrieval", {})
    mode = mode or rcfg.get("mode", "full")
    result = _new_result(pitch, cfg, mode)
    try:
        _run(result, pitch, catalog, cfg, cache, mode, int(k or rcfg.get("k", 5)))
    except ProviderError as e:
        result.update(status="provider_error", status_reason=e.kind, verdict=None, provider_error=e.to_dict())
    except Exception as e:  # a defect in this tool, not in the model or the provider
        result.update(status="internal_error", status_reason=type(e).__name__, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    result["seconds"] = round(time.time() - t0, 2)
    return result


def _run(result, pitch, catalog, cfg, cache, mode, k):
    if not isinstance(pitch, str) or not pitch.strip():
        result.update(status="invalid_input", status_reason="empty_pitch", error="The pitch is empty.")
        return
    if len(pitch) > MAX_PITCH_CHARS:
        result.update(status="invalid_input", status_reason="pitch_too_long", error=f"The pitch is longer than {MAX_PITCH_CHARS} characters; a 2-4 sentence pitch is expected.")
        return

    by_id = {r["id"]: r for r in catalog}
    baseline = rank(pitch, {r["id"]: r["synopsis"] for r in catalog})
    result["baseline_top"] = [{"id": i, "score": round(s, 3)} for i, s in baseline[:5]]

    cards = {}
    for r in catalog:
        cards[r["id"]] = cache.get(r["synopsis"], cfg, "catalog")
        if not cards[r["id"]]["well_formed"] or not cards[r["id"]]["events"]:
            result["catalog_warnings"].append(f"{r['id']}: the model's event list for this synopsis was unusable; it is compared from its synopsis text only")

    pcard = extract_card(pitch, cfg, "pitch")
    result["raw"]["pitch_extraction_reply"] = pcard["raw"]
    result["pitch_extraction"] = {k_: v for k_, v in pcard.items() if k_ != "raw"}
    state, why = assess_pitch(pitch, pcard)
    stats = {"events": len(pcard["events"]), "words": word_count(pitch)}

    if state == "unusable":
        result.update(status="unverified", status_reason="extraction_unusable", verdict=None,
                      error=f"No comparison was made: {why}. This is a model failure, not a finding about the pitch. The raw reply is kept.")
        return

    if state == "vague":
        report = {"verdict": "insufficient_information", "closest_id": None, "shared_events": [], "first_difference": None,
                  "order_basis": None, "editor_question": follow_up_question(pcard),
                  "missing_info": [f"not stated in the pitch: {', '.join(pcard['missing_roles']) or 'enough events'}"]}
        checks = check_report(report, pitch, by_id, stats)
        result.update(status="verified" if checks["ok"] else "internal_error", status_reason="code_gate_thin_pitch",
                      verdict="insufficient_information", claimed_verdict="insufficient_information", report=report, checks=checks,
                      decided_by=f"code gate: valid extraction found too little to compare ({why})")
        return

    if mode == "shortlist":
        candidates = [by_id[i] for i, _ in baseline[:k]]
    else:
        candidates = list(catalog)
    result["candidates"] = [c["id"] for c in candidates]

    prompt = build_compare_prompt(pitch, pcard, candidates, cards)
    ctx = {"pitch": {"text": pitch, "events": pcard["events"]},
           "catalog": [{"id": c["id"], "text": c["synopsis"], "events": cards[c["id"]]["events"]} for c in candidates]}
    result["raw"]["compare_prompt"] = prompt
    raw = complete(prompt, cfg, task="compare", context=ctx)
    result["raw"]["compare_reply"] = raw
    try:
        report = parse_json(raw)
    except ModelOutputError as e:
        result.update(status="unverified", status_reason="reply_not_json", error=f"{e}. The raw reply is kept.")
        return
    report, checks, repairs = _verify(report, pitch, candidates, stats)
    result["report"] = report
    result["checks"] = checks
    result["quote_repairs"] = repairs
    claimed = as_dict(report).get("verdict")
    result["claimed_verdict"] = claimed if claimed in VERDICTS else None

    if not checks["ok"]:
        # One feedback retry: the model sees its own reply and the failed checks. Both replies are kept.
        fails = "\n".join("- " + f for f in checks["failures"][:6])
        result["retry"] = {"used": True, "first_failures": list(checks["failures"]), "outcome": None}
        rprompt = prompt + "\n\nYOUR PREVIOUS REPLY:\n" + raw + "\n\n" + prompts.RETRY.format(failures=fails)
        result["raw"]["retry_reply"] = None
        try:
            rraw = complete(rprompt, cfg, task="compare", context=ctx)
        except ProviderError as e:
            rraw = None
            result["retry"]["outcome"] = f"retry could not be sent ({e.kind}); first reply kept"
        result["raw"]["retry_reply"] = rraw
        report2 = None
        if rraw is not None:
            try:
                report2 = parse_json(rraw)
            except ModelOutputError:
                result["retry"]["outcome"] = "retry reply was not JSON; first reply kept"
        if report2 is not None:
            report2, checks2, repairs2 = _verify(report2, pitch, candidates, stats)
            if checks2["ok"]:
                report, checks, repairs = report2, checks2, repairs2
                checks["warnings"].append("This result comes from a second attempt: the first reply failed the source checks and the model was asked once to correct it.")
                result.update(report=report, checks=checks, quote_repairs=repairs)
                result["claimed_verdict"] = report.get("verdict") if report.get("verdict") in VERDICTS else None
                result["retry"]["outcome"] = "retry passed the checks"
            else:
                result["retry"]["outcome"] = "retry still failed the checks; first reply kept"
                result["retry"]["second_failures"] = list(checks2["failures"])

    if checks["ok"]:
        used_retry = bool(result.get("retry"))
        result.update(status="verified", status_reason="checks_passed_after_retry" if used_retry else "checks_passed", verdict=report["verdict"])
    else:
        result.update(status="unverified", status_reason="report_failed_checks", verdict=None)


# --------------------------------------------------------------------------- plain-text view (CLI)

def format_text(result, catalog=None):
    by_id = {r["id"]: r for r in (catalog or [])}
    L = []
    if result.get("mock"):
        L.append("*** MOCK PROVIDER: offline stand-in, NOT a model result. Set a real provider in config.json. ***")
    st = result.get("status")
    if st == "provider_error":
        pe = as_dict(result.get("provider_error"))
        L += [f"PROVIDER ERROR ({pe.get('kind')}): {pe.get('message')}", f"What to do: {pe.get('hint')}", "No comparison was made."]
        return "\n".join(L)
    if st in ("internal_error", "invalid_input"):
        L.append(f"{st.upper().replace('_', ' ')}: {result.get('error') or result.get('status_reason')}")
        return "\n".join(L)
    rep = as_dict(result.get("report"))
    if st == "unverified":
        L.append("UNVERIFIED: do not rely on this result. " + as_str(result.get("error")))
        if result.get("claimed_verdict"):
            L.append(f"The model claimed: {result['claimed_verdict']} (unverified)")
    else:
        L.append(f"VERDICT: {result.get('verdict')}   [source checks passed]")
        if result.get("decided_by"):
            L.append(f"({result['decided_by']})")
    cid = rep.get("closest_id")
    if isinstance(cid, str) and cid:
        row = by_id.get(cid, {})
        L.append(f"Closest show: {row.get('title', cid)} ({cid})")
        if row.get("source_url"):
            L.append(f"Source to inspect: {row['source_url']}")
    for i, ev in enumerate(as_list(rep.get("shared_events")), 1):
        ev = as_dict(ev)
        L.append(f"  {i}. pitch text:                  \"{as_str(ev.get('pitch_excerpt'), 300)}\"")
        L.append(f"     Evidence from catalog synopsis: \"{as_str(ev.get('catalog_excerpt'), 300)}\"")
        if ev.get("what_is_shared"):
            L.append(f"     shared: {as_str(ev['what_is_shared'], 300)}")
    fd = as_dict(rep.get("first_difference"))
    if fd:
        head = "First supported difference" if fd.get("established") is True else "First difference NOT established by the sources"
        L.append(f"{head}: {as_str(fd.get('explanation'), 400)}")
    if rep.get("editor_question"):
        L.append(f"Question for the editor: {as_str(rep['editor_question'], 300)}")
    c = as_dict(result.get("checks"))
    if c:
        L.append(f"Mechanical checks: {c.get('quotes_verified', 0)}/{c.get('quotes_checked', 0)} quoted passages found in the source text")
        for f in as_list(c.get("failures")):
            L.append(f"  FAILED: {f}")
        for w in as_list(c.get("warnings")):
            L.append(f"  note: {w}")
    return "\n".join(L)


MATCH_VERDICT_SET = MATCH_VERDICTS  # re-exported for the renderer
