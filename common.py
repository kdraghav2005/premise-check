import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pitchcheck.pipeline import load_catalog  # noqa: E402

CATALOG = load_catalog(os.path.join(HERE, "data", "catalog.csv"))
BY_ID = {r["id"]: r for r in CATALOG}

# A detailed, clear pitch (>= 25 words, several events)
CLEAR_PITCH = (
    "After her fiance leaves her for another designer, a wedding photographer immediately marries a stranger. "
    "She learns his studio is the main rival of her ex's firm, and the two of them compete for a major contract."
)
# A genuinely vague pitch
VAGUE_PITCH = "A photographer falls for a billionaire."

MOCK_CFG = {"provider": "mock", "model": "mock", "retrieval": {"mode": "full", "k": 3}}


def good_report():
    """A report that is valid against SHOW_001 and CLEAR_PITCH."""
    return {
        "verdict": "same_sequence",
        "closest_id": "SHOW_001",
        "shared_events": [
            {"pitch_excerpt": "leaves her for another designer", "catalog_excerpt": "husband leaves her for a woman impersonating",
             "what_is_shared": "her partner leaves her for another woman", "kind": "opening_conflict",
             "pitch_agency": "other_acts", "catalog_agency": "other_acts"},
            {"pitch_excerpt": "immediately marries a stranger", "catalog_excerpt": "quickly marries Roy",
             "what_is_shared": "she quickly marries someone new", "kind": "decision",
             "pitch_agency": "protagonist_chooses", "catalog_agency": "protagonist_chooses"},
            {"pitch_excerpt": "his studio is the main rival of her ex's firm", "catalog_excerpt": "he is her ex's business rival",
             "what_is_shared": "she learns the new husband is her ex's rival", "kind": "reveal",
             "pitch_agency": "discovery", "catalog_agency": "discovery"},
        ],
        "order_basis": "stated",
        "first_difference": {"established": True, "explanation": "The synopsis ends with joining him to compete; the pitch does not say whether she joins him.",
                             "pitch_excerpt": "the two of them compete for a major contract", "catalog_excerpt": "joins him to compete for a technology contract"},
        "editor_question": "Does the photography setting change what she decides after the rival reveal?",
        "missing_info": [],
    }
