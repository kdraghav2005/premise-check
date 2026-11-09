import re
import sys
import unicodedata

STOP = set(
    "a an the and or but of to in on at for with by from as is are was were be been it its "
    "his her their he she they them that this who whom which after before then when while into "
    "out up down over under again him not no has have had so than too very can will just own "
    "one two".split()
)

MIN_EVIDENCE_CHARS = 12  # excerpts shorter than this are not accepted as evidence


def setup_stdout():
    """Windows consoles often default to a legacy code page; never crash on curly quotes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
    )
    s = re.sub("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad]", "-", s)  # every hyphen/dash variant (a live model wrote U+2011)
    return re.sub(r"\s+", " ", s).strip().lower()


def find_range(excerpt, text):
    """(start, end) of excerpt inside text after light normalisation, or None.
    Positions refer to the normalised text, so they are comparable across excerpts of the same text."""
    e = norm(excerpt).strip(" .…\"'")
    if len(e) < MIN_EVIDENCE_CHARS:
        return None
    i = norm(text).find(e)
    return (i, i + len(e)) if i >= 0 else None


def find_span(excerpt, text):
    r = find_range(excerpt, text)
    return r[0] if r else -1


def word_count(s):
    return len(re.findall(r"\w+", s or ""))


def tokens(text):
    return [w for w in re.findall(r"[a-z]+", norm(text)) if w not in STOP and len(w) > 2]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def closest_span(excerpt, text, max_word_diff=2, min_ratio=0.85):
    """Source wording for a quote that is almost, but not exactly, in `text`.
    Returns the exact slice of `text` when a window of whole words differs from the quote by at most `max_word_diff`
    words (a dropped or swapped small word, as in a live run: "turns out be" for "turns out to be"), else None.
    A passage that is not in the text at all (an invented sentence) has no close window and returns None."""
    import difflib

    def clean(w):
        return norm(w).strip(".,;:!?\"'()[]…")

    want = [clean(w) for w in (excerpt or "").split()]
    want = [w for w in want if w]
    if len(want) < 3:
        return None
    spans = [(m.start(), m.end(), clean(m.group())) for m in re.finditer(r"\S+", text or "")]
    spans = [s for s in spans if s[2]]
    words = [s[2] for s in spans]
    best = None
    for n in range(max(3, len(want) - max_word_diff), len(want) + max_word_diff + 1):
        for i in range(0, len(words) - n + 1):
            sm = difflib.SequenceMatcher(None, want, words[i:i + n], autojunk=False)
            ratio = sm.ratio()
            if ratio < min_ratio:
                continue
            diff = sum(max(j2 - j1, i2 - i1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
            if diff > max_word_diff:
                continue
            if best is None or (ratio, -diff) > (best[0], -best[1]):
                best = (ratio, diff, i, i + n)
    if best is None:
        return None
    _, _, a, b = best
    return text[spans[a][0]:spans[b - 1][1]].strip(" ,;:")
