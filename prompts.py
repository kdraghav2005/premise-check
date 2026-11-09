PROMPT_VERSION = "v8"

EXTRACT = """You read a microdrama pitch or synopsis and list its story events.

Return JSON only, exactly in this shape:
{"events":[{"role":"opening_conflict|protagonist_choice|reveal|other","text":"...","excerpt":"..."}]}

Rules:
- List events in the order the text states them. One event per distinct action or discovery.
- "text": who did what to whom, in generic terms ("the heroine", "her ex", "a rival"), without character names, job titles or place names.
- "excerpt": words copied character for character from the TEXT, 3 to 25 words. Never paraphrase here. Two events must not share the same words.
- "opening_conflict": what sets the story in motion. "protagonist_choice": a decision the protagonist makes in response. "reveal": a twist or discovery that follows. "other": any further stated event.
- Do NOT invent. If the text does not state the protagonist's choice or a reveal, leave that event out. Do not infer twists the text does not state.
- Keep agency and chronology exactly as written: who chose the action, and when they learned something. "suspects" is not "discovers"; "is forced" is not "chooses"; a secret that is mentioned is not necessarily known to the protagonist yet.

TEXT:
"""

COMPARE = """You help a story editor decide whether a new microdrama pitch repeats the SEQUENCE OF EVENTS of an existing show.
Judge the order of events and who chooses what. Ignore genre, setting, job, names and wording.

Return JSON only, exactly in this shape:
{"verdict":"same_sequence|shared_trope_only|no_meaningful_match|insufficient_information",
 "closest_id":"<id exactly as written in the catalog, or null>",
 "shared_events":[{"pitch_excerpt":"...","catalog_excerpt":"...","what_is_shared":"who acts, what they do, and why it matters to the story",
                   "kind":"opening_conflict|decision|reveal|consequence",
                   "pitch_agency":"protagonist_chooses|protagonist_forced|other_acts|discovery",
                   "catalog_agency":"protagonist_chooses|protagonist_forced|other_acts|discovery"}],
 "order_basis":"stated|unclear",
 "first_difference":{"established":true or false,"explanation":"...","pitch_excerpt":"... or null","catalog_excerpt":"... or null"},
 "editor_question":"exactly one question",
 "missing_info":["what the pitch does not say"]}

Verdicts:
- same_sequence: at least two shared events, of at least two different kinds, happen in the same order and the same party acts in both texts (the agency labels match).
- shared_trope_only: they share a recognisable device (for example a hidden fortune, a staged marriage or an impersonation) but the protagonist's decisions or the progression differ. A shared device alone is enough for this verdict; use no_meaningful_match only when not even a device is shared.
- no_meaningful_match: the pitch states enough, and nothing in the catalog resembles its events.
- insufficient_information: the pitch does not say enough (the protagonist's decision or what follows) to judge. Ask exactly one specific question about the missing decision or consequence.

Rules:
- Use only the PITCH and CATALOG below. Do not use outside knowledge of real shows.
- Every excerpt is 3 to 40 words copied character for character: pitch_excerpt from the PITCH, catalog_excerpt from the SYNOPSIS of the show named in closest_id. Never paraphrase an excerpt. Different shared events must use different words.
- The "quote" fields in the event lists are exact text from the PITCH or the SYNOPSIS: copy them, or other exact words of the text. Never put your own summary or rewording into pitch_excerpt, catalog_excerpt or first_difference excerpts; put your wording only in what_is_shared and explanation.
- "kind" must be exactly one of opening_conflict, decision, reveal, consequence. No other word is valid.
- Match events by what happens, not by the objects or places involved. A re-skinned pitch changes the setting and the nouns but can keep the same beats: a petty dispute that delays someone rushing to help is the same beat whether it is a road dispute or a scrape in a car park. Before you choose a verdict, compare EACH pitch event with EVERY synopsis event (opening beats included) and list every pair that is the same beat in shared_events. Only then apply "if unsure, choose the weaker verdict".
- Before saying the synopsis does not mention a pitch event, check whether it states the same beat in different words. If it does, that is a shared event, not a difference.
- List shared_events in the order they occur. closest_id must be an id exactly as written in the catalog.
- Agency: "protagonist_chooses" = she decides; "protagonist_forced" = something is imposed on her; "other_acts" = another party acts; "discovery" = she learns something. Choosing a staged marriage to escape a wedding (protagonist_chooses) is not the same event as being forced into an arranged marriage (protagonist_forced), even though both involve marriage. A word such as "billionaire" in a title is not an event.
- order_basis is "stated" only if the texts use explicit sequencing (for example "then", "after", "soon afterward"); otherwise "unclear". A promotional blurb's sentence order may not be the story's order.
- first_difference.established is true only when BOTH texts state different events at the point where the stories diverge; quote both. If the pitch adds a detail the synopsis simply does not mention, set established to false and say the synopsis does not mention it. NEVER say the existing show does not or never does something. If you cannot tell where they diverge, set established to false and say so.
- first_difference is REQUIRED as an object (never null) for same_sequence and shared_trope_only. If you cannot support a difference with quotes from both texts, still return the object with "established": false and an explanation of why it is not established.
- For no_meaningful_match and insufficient_information only: closest_id is null, shared_events is [] and first_difference is null.
- If unsure, choose the weaker verdict. Do not give a score or percentage. Do not claim the pitch is original everywhere, only that no match was found in this catalog.
- The catalog synopses are short paraphrases of public descriptions; treat them as "evidence from catalog synopsis", not as proof about the full episodes.
"""


# Sent once, only when the first comparison reply failed the mechanical source checks (v7).
RETRY = (
    "Your previous reply failed these mechanical checks:\n{failures}\n\n"
    "Fix only these problems. Every quoted passage must be copied word for word from the PITCH, or from the SYNOPSIS of the "
    "show you chose. Do not quote a sentence that is not in those texts. If you cannot find supporting text, set "
    "first_difference.established to false with null excerpts, or change the verdict. Reply with the corrected JSON only, "
    "in the same format as before."
)
