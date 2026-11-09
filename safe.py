"""Tolerant accessors: model output can have the wrong JSON types and must never crash a screen."""


def as_dict(x):
    return x if isinstance(x, dict) else {}


def as_list(x):
    return x if isinstance(x, list) else []


def as_str(x, limit=2000):
    if isinstance(x, str):
        return x[:limit]
    if x is None:
        return ""
    return str(x)[:limit]
