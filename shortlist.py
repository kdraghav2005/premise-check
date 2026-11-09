"""Keyword baseline: TF-IDF cosine ranking in pure Python.

Used (a) as the baseline the AI comparison must beat and (b) as an optional
retrieval step for large catalogs (mode "shortlist"). Never used in "full" mode.
"""
import math
from collections import Counter

from .textutil import tokens


def rank(query, docs):
    """docs: {id: text}. Returns [(id, score)] sorted best-first."""
    toks = {i: tokens(t) for i, t in docs.items()}
    n = len(docs)
    df = Counter()
    for ts in toks.values():
        df.update(set(ts))
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
    default_idf = math.log(1 + n) + 1

    def vec(ts):
        c = Counter(ts)
        return {w: (1 + math.log(k)) * idf.get(w, default_idf) for w, k in c.items()}

    def cos(a, b):
        dot = sum(v * b.get(w, 0.0) for w, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    q = vec(tokens(query))
    scored = [(i, cos(q, vec(ts))) for i, ts in toks.items()]
    return sorted(scored, key=lambda x: (-x[1], x[0]))
