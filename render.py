"""HTML for the editor's screen. Everything that comes from the pitch, the catalog or the model is escaped.
Model output may have the wrong JSON types, so every field goes through the tolerant accessors."""
import html
import json
import re

from .safe import as_dict, as_list, as_str
from .verify import MATCH_VERDICTS

VERDICT_TITLES = {
    "same_sequence": "Meaningful sequence overlap",
    "shared_trope_only": "Shared trope only",
    "no_meaningful_match": "No meaningful match in this catalog",
    "insufficient_information": "Insufficient information",
}
VERDICT_BLURB = {
    "same_sequence": "Several consequential events align, in the same order, with the same party acting.",
    "shared_trope_only": "A familiar device is shared, but the protagonist's choices or the central conflict differ.",
    "no_meaningful_match": "Nothing in the catalog that was searched supports a meaningful overlap. This says nothing about shows outside it.",
    "insufficient_information": "The pitch does not state enough to compare. Answer the question below and run it again.",
}
AGENCY_TEXT = {
    "protagonist_chooses": "she chooses", "protagonist_forced": "it is imposed on her",
    "other_acts": "another party acts", "discovery": "she discovers it",
}

CSS = """
body{font-family:system-ui,Segoe UI,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 1rem;color:#1f2328;line-height:1.45}
h1{margin-bottom:.2rem} .sub{color:#57606a;margin-top:0}
textarea{width:100%;height:8rem;font:inherit;padding:.6rem;box-sizing:border-box;border:1px solid #8c959f;border-radius:6px}
button{padding:.55rem 1.4rem;font:inherit;cursor:pointer;background:#0969da;color:#fff;border:0;border-radius:6px}
.mock{background:#ffe08a;border:3px solid #b8860b;padding:.8rem 1rem;border-radius:8px;margin:1rem 0;font-weight:600}
.card{border:1px solid #d0d7de;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0;background:#fff}
.verdict{font-size:1.5rem;font-weight:700;margin:.2rem 0}
.chip{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.8rem;font-weight:700;color:#fff}
.ok{background:#1a7f37}.warn{background:#bc4c00}.bad{background:#cf222e}.grey{background:#6e7781}
.banner{padding:.7rem 1rem;border-radius:8px;margin:1rem 0}
.b-bad{background:#ffebe9;border:1px solid #cf222e}.b-warn{background:#fff1e5;border:1px solid #bc4c00}.b-grey{background:#f0f2f4;border:1px solid #8c959f}
.ev{border-top:1px solid #eaeef2;padding:.7rem 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
.q{background:#f6f8fa;border-left:4px solid #8c959f;padding:.5rem .7rem;border-radius:4px}
.q b{display:block;font-size:.75rem;text-transform:uppercase;color:#57606a;margin-bottom:.2rem}
.note{color:#57606a;font-size:.9rem}
.question{background:#ddf4ff;border:1px solid #54aeff;border-radius:8px;padding:.7rem 1rem;font-weight:600}
details{margin:.6rem 0} summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:.8rem;border-radius:6px;font-size:.8rem;max-height:24rem;overflow:auto}
li.f{color:#cf222e} li.p{color:#1a7f37}
@media(max-width:700px){.pair{grid-template-columns:1fr}}
"""


def esc(x):
    return html.escape(as_str(x), quote=True)


def safe_url(u):
    u = as_str(u).strip()
    return u if re.match(r"^https?://[^\s<>\"']+$", u) else None


def _quote(label, text):
    return f'<div class="q"><b>{esc(label)}</b>{esc(text) if as_str(text) else "<i>(none given)</i>"}</div>'


def _mock_banner(result_or_cfg):
    return (
        '<div class="mock">MOCK PROVIDER. This screen is running an offline stand-in, not a language model. '
        "Its output is a plumbing check and says nothing about how the tool performs. Set a real provider in config.json.</div>"
    )


def _status_block(result):
    st = result.get("status")
    if st == "provider_error":
        pe = as_dict(result.get("provider_error"))
        return (
            '<div class="banner b-warn"><span class="chip warn">PROVIDER ERROR</span> '
            f"<b>No comparison was made.</b> {esc(pe.get('message'))}"
            f"<br><b>What to do:</b> {esc(pe.get('hint'))}"
            '<br><span class="note">This is not a finding about your pitch. Nothing was compared.</span></div>'
        )
    if st == "unverified":
        reason = {
            "extraction_unusable": "The model could not turn the pitch into usable events.",
            "reply_not_json": "The model's comparison was not valid JSON.",
            "report_failed_checks": "The model's comparison failed the mechanical source checks.",
        }.get(result.get("status_reason"), "The result could not be verified.")
        return (
            '<div class="banner b-bad"><span class="chip bad">UNVERIFIED</span> '
            f"<b>Do not rely on this result.</b> {esc(reason)} {esc(result.get('error'))}"
            '<br><span class="note">The model\'s raw reply is preserved under Technical details and in the download.</span></div>'
        )
    if st == "invalid_input":
        return f'<div class="banner b-grey"><span class="chip grey">INVALID INPUT</span> {esc(result.get("error"))}</div>'
    if st == "internal_error":
        return (
            '<div class="banner b-grey"><span class="chip grey">TOOL ERROR</span> Something went wrong inside this tool, '
            f"not in the model or provider: {esc(result.get('error'))}. The run was saved; download it to share the details.</div>"
        )
    return ""


def _verdict_block(result):
    st = result.get("status")
    if st == "verified":
        v = result.get("verdict")
        chip = '<span class="chip ok">SOURCE CHECKS PASSED</span>'
        return (
            f'<div class="verdict">{esc(VERDICT_TITLES.get(v, v))} {chip}</div>'
            f'<div class="note">{esc(VERDICT_BLURB.get(v, ""))}</div>'
            + (f'<div class="note">{esc(result.get("decided_by"))}</div>' if result.get("decided_by") else "")
        )
    claimed = result.get("claimed_verdict")
    if st == "unverified" and isinstance(claimed, str) and claimed in VERDICT_TITLES:
        return f'<div class="note">What the model claimed (unverified, treat as a guess):</div><div class="verdict" style="opacity:.6">{esc(VERDICT_TITLES[claimed])}</div>'
    return ""


def _shared_events(rep, show_all):
    events = as_list(rep.get("shared_events"))
    if not events:
        return ""
    out = ["<h3>Shared events, in order</h3>"]
    for i, ev in enumerate(events, 1):
        ev = as_dict(ev)
        pa, ca = as_str(ev.get("pitch_agency")), as_str(ev.get("catalog_agency"))
        differs = pa and ca and pa != ca
        who = f"In your pitch: {AGENCY_TEXT.get(pa, pa)}. In the synopsis: {AGENCY_TEXT.get(ca, ca)}." if (pa or ca) else ""
        out.append(
            '<div class="ev">'
            f"<b>{i}. {esc(ev.get('kind', 'event')).replace('_', ' ')}</b>"
            f'<div class="pair">{_quote("From your pitch", ev.get("pitch_excerpt"))}{_quote("Evidence from catalog synopsis", ev.get("catalog_excerpt"))}</div>'
            f'<div class="note">{esc(ev.get("what_is_shared"))}</div>'
            + (f'<div class="note"><b>{"Who acts differs. " if differs else ""}</b>{esc(who)}</div>' if who else "")
            + "</div>"
        )
    return "".join(out)


def _difference(rep):
    fd = as_dict(rep.get("first_difference"))
    if not fd:
        return ""
    if fd.get("established") is True:
        return (
            "<h3>First supported difference</h3>"
            f"<div>{esc(fd.get('explanation'))}</div>"
            f'<div class="pair">{_quote("From your pitch", fd.get("pitch_excerpt"))}{_quote("Evidence from catalog synopsis", fd.get("catalog_excerpt"))}</div>'
        )
    return (
        "<h3>First difference: not established by the sources</h3>"
        f"<div>{esc(fd.get('explanation'))}</div>"
        '<div class="note">A detail missing from a short blurb does not show that the existing show lacks it. '
        "Read the full show before treating it as a difference.</div>"
        + (f'<div class="pair">{_quote("From your pitch", fd.get("pitch_excerpt"))}{_quote("Evidence from catalog synopsis", fd.get("catalog_excerpt"))}</div>' if (fd.get("pitch_excerpt") or fd.get("catalog_excerpt")) else "")
    )


def _checks(result):
    c = as_dict(result.get("checks"))
    if not c:
        return ""
    lis = "".join(
        f'<li class="{"p" if as_dict(i).get("passed") else "f"}">{"passed" if as_dict(i).get("passed") else "FAILED"}: {esc(as_dict(i).get("label"))}'
        + (f" ({esc(as_dict(i).get('detail'))})" if as_dict(i).get("detail") else "")
        + "</li>"
        for i in as_list(c.get("items"))
    )
    warns = "".join(f"<li>{esc(w)}</li>" for w in as_list(c.get("warnings")))
    return (
        f"<h4>Mechanical checks: {esc(c.get('quotes_verified', 0))}/{esc(c.get('quotes_checked', 0))} quoted passages found in the source text</h4>"
        f'<div class="note">{esc(c.get("scope_note"))}</div><ul>{lis}</ul>'
        + (f"<h4>Notes for the editor</h4><ul>{warns}</ul>" if warns else "")
    )


def render_result(result, catalog):
    """HTML for one run (any status)."""
    by_id = {r["id"]: r for r in catalog}
    rep = as_dict(result.get("report"))
    st = result.get("status")
    parts = ['<div class="card">']
    if result.get("mock"):
        parts.append(_mock_banner(result))
    parts.append(_status_block(result))
    parts.append(_verdict_block(result))

    show_report = st in ("verified", "unverified") and rep
    cid = rep.get("closest_id") if show_report else None
    if isinstance(cid, str) and cid:
        row = by_id.get(cid)
        if row:
            url = safe_url(row.get("source_url"))
            link = f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">open source</a>' if url else '<span class="note">(no source link in the catalog)</span>'
            parts.append(f"<h3>Closest show</h3><div><b>{esc(row.get('title'))}</b> <span class=\"note\">({esc(cid)})</span> &middot; {link}</div>")
        else:
            parts.append(f'<h3>Closest show</h3><div class="note">The model named {esc(cid)}, which is not in the catalog.</div>')
    if show_report:
        parts.append(_shared_events(rep, st))
        parts.append(_difference(rep))
        q = as_str(rep.get("editor_question")).strip()
        if q:
            parts.append(f'<h3>{"Question to answer first" if rep.get("verdict") == "insufficient_information" else "Question for the editor"}</h3><div class="question">{esc(q)}</div>')
        mi = [as_str(m) for m in as_list(rep.get("missing_info")) if m]
        if mi:
            parts.append('<div class="note">Not stated in the pitch: ' + esc("; ".join(mi)) + "</div>")

    # technical details behind a disclosure
    tech = [f'<div class="note">provider / model: {esc(result.get("provider"))} / {esc(result.get("model"))} &middot; mode: {esc(result.get("mode"))} &middot; '
            f'run {esc(result.get("run_id"))} &middot; {esc(result.get("seconds"))} s</div>']
    tech.append(_checks(result))
    for w in as_list(result.get("catalog_warnings")):
        tech.append(f'<div class="note">{esc(w)}</div>')
    raw = json.dumps(result, ensure_ascii=False, indent=1)
    tech.append(f'<details><summary>Raw run JSON (pitch, replies, checks, status)</summary><pre>{esc(raw)}</pre></details>')
    parts.append(f'<details><summary>Technical details</summary>{"".join(tech)}</details>')
    parts.append(f'<p><a href="/download/{esc(result.get("run_id"))}.json" download>Download this run (JSON)</a></p>')
    parts.append("</div>")
    return "".join(parts)


def render_page(cfg, catalog, pitch="", result=None):
    banner = _mock_banner(cfg) if str(cfg.get("provider", "")).startswith("mock") else ""
    body = render_result(result, catalog) if result else ""
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Premise Check</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Premise Check</h1>"
        f'<p class="sub">Paste a 2-4 sentence pitch. It is compared with all {len(catalog)} shows in the catalog. '
        "The tool flags a resemblance and shows the evidence; the editor decides.</p>"
        f"{banner}"
        '<form method="post" action="/check"><textarea name="pitch" required maxlength="4000" placeholder="Paste a pitch...">'
        f"{esc(pitch)}</textarea>"
        f'<p><button type="submit">Compare</button> <span class="note">model: {esc(cfg.get("provider"))} / {esc(cfg.get("model"))}</span></p></form>'
        f"{body}</body></html>"
    )
