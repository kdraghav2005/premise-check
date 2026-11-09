# Premise Check (kuku_pitchcheck)

A pitch-review helper for a story editor. Paste a 2-4 sentence microdrama pitch; the tool compares it with **all 10 shows** in
`data/catalog.csv` and shows the closest show, the shared story events (each with a quote from the pitch and a quote from the
synopsis), the first supported difference, and one question for the editor. The editor decides. Pure Python standard library; no installs.

**Prompt version in this package: v8** (v7 plus one rule in the comparison prompt; its dev and holdout results are in the table below). Default `config.json` is a **mock** provider (offline, no model; the page says so in a banner).
No API key or private file is included.

## Run on Windows (PowerShell)
```powershell
cd path\to\kuku_pitchcheck
py -3 --version                                   # Python 3.9 or newer
py -3 -m unittest discover -s tests               # 93 offline tests
py -3 server.py                                   # mock mode: opens at http://127.0.0.1:8000
```
Live model (Groq free tier; any OpenAI-compatible API works by changing `base_url` and `model`):
```powershell
Copy-Item config.groq.example.json config.json    # model: openai/gpt-oss-120b (check your account's models first)
$sec = Read-Host "Paste Groq key (hidden)" -AsSecureString
$env:PITCHCHECK_API_KEY = [System.Net.NetworkCredential]::new("", $sec).Password
py -3 server.py                                   # banner must show provider openai_compat, no MOCK line
```
List the models your key can use: `(Invoke-RestMethod -Uri "https://api.groq.com/openai/v1/models" -Headers @{Authorization="Bearer $env:PITCHCHECK_API_KEY"}).data.id | Sort-Object`.
The key is read only from the environment variable; it is never written to a file or a run record.
The first run reads each catalog show once (cached in `outputs\`); on the free tier this can take over a minute because of the per-minute token limit.

CLI for one pitch: `py -3 run.py --pitch-file mypitch.txt` (exit 0 verified, 2 unverified, 3 provider error, 4 invalid input).

## Outcomes
- Editorial verdict (only when verified): `same_sequence`, `shared_trope_only`, `no_meaningful_match`, `insufficient_information`.
- Technical status: `verified`, `unverified` (the model's reply failed the source checks; raw reply kept, treat as a guess), `provider_error`, `invalid_input`, `internal_error`.
- Python checks that the show id exists, each quote occurs in the pitch or in the chosen synopsis, quotes are distinct and in the same text order, and field types/labels are valid.
  It does **not** check that the events mean the same thing or that the editorial call is right. A person reads the paired quotes.
- Every run (including failures) can be downloaded as JSON from the result page and is saved in `outputs\runs\`.

## Evaluation (`data/test_cases.csv`: 12 authored pitches, 7 dev + 5 holdout)
```powershell
py -3 eval.py                  # dev split; raw outputs and summary.md in outputs\eval\<run>\
py -3 eval.py --freeze         # record prompt/code/config/catalog/label hashes
py -3 eval.py --split holdout  # 5 cases, once per freeze
```
The model never receives labels. The keyword baseline (`pitchcheck/shortlist.py`) is graded on retrieval only.

Naive baseline (`naive_baseline.py`, not part of the app): the same model gets one plain prompt ("is this a repeat? quote your evidence") with no event extraction, no verification and no retry, on all 12 cases, once. Afterwards the same source check is applied only to measure how often its quotes are not in the text. The prompt was written once and is not tuned. It never touches the freeze record or holdout log.
```powershell
py -3 naive_baseline.py        # needs a real provider in config.json and the key in the environment
```
Result (one run, 12 cases, same model, nothing tuned): label agreement **7/12** against Premise Check v8's 11/12 (holdout alone: 3/5 against 5/5). The plain prompt's quotes were mostly real (11 of 12 quoted passages found in the source), so on this model the quote check mattered less than expected. Its costly error was a *false all-clear*: it said `no_meaningful_match` for 4 pitches that share a device or sequence with a catalog show (T02, T04, T05, T06), which v8 never did (0/12); it also downgraded T12 to `shared_trope_only`. Caveat: v8's prompts were tuned on the 7 dev cases and the baseline was not, so the holdout is the fairer comparison, and 5 cases is small.

Measured with Groq `openai/gpt-oss-120b` (temperature 0), one run each:

| Prompt / split | Label agreement | Verified success | Quoted passages found |
|---|---|---|---|
| v5, dev (7 cases) | 4/7 | 3/7 | 21/22 |
| v6, dev (tuned on these same 7 cases) | 6/7 | 5/7 | 23/25 |
| v7, dev (its quote-correction and retry handlers never triggered) | 6/7 | 6/7 | 25/25 |
| v8, dev (this package; tuned on these same 7 cases) | 6/7 | 6/7 | 23/23 |
| **v8, holdout (5 cases, run once, freeze `c8be4a9cd988`)** | **5/5** | **5/5** | **16/16** |

Read these as behaviours and failures, not production accuracy: 12 authored cases, one run each, and results can differ between runs even at temperature 0 (T04 was unverified in v6 and verified in v7 and v8 although the code that handles it never fired). The quote-correction and retry handlers were not triggered in any live run, so they are tested only offline. The holdout was run once and its result is reported as it came out; the keyword baseline also ranked the intended show first on both holdout retrieval cases (2/2), so what the model adds is the verdicts and quoted evidence.

Failures seen in the live runs:
- T02 (dev, v6): the model dropped one word when quoting a synopsis (`turns out be` for `turns out to be`), so the run was unverified. v7 corrects a quote that is at most two words off to the exact source wording and shows a "Quote corrected" note with both versions.
- T04 (dev, v6): the model quoted a sentence that is not in the synopsis at all. This is never corrected. v7 makes one retry that shows the model its own reply and the failed checks; a result that passes only after the retry says so, and both replies are saved. A second failure stays unverified.
- **T02 (dev, v7 and v8) remains a wrong label:** the model says `shared_trope_only`, the key says `same_sequence`. Its quotes verify. It pairs only the reveal and does not pair "confiscate her keys" (the pitch) with "delays a doctor ... after a road dispute" (the synopsis), although both are the same beat in different words. The v8 prompt rule added to fix this did not change the answer, and no further tuning was done.

**Added after the evaluations above (not part of the evaluated code):** `pitchcheck/llm.py` now retries up to twice when Groq answers HTTP 400 `json_validate_failed` ("max completion tokens reached before generating a valid document"), which happened once in a manual run of the web page when the model's reasoning used up its output budget. If it still fails, the page shows a provider error with a plain hint. The prompts, verifier, catalog and model settings are unchanged. No evaluation was re-run after this change, so the code hash differs from the freeze record (`c8be4a9cd988`).

## Layout
`server.py` web UI, `run.py` CLI, `eval.py` evaluation, `pitchcheck/` (llm adapters, prompts, verify, render), `data/` (catalog of 10 public show synopses with source links; 12 test pitches), `tests/`.
`data/catalog.csv` holds short paraphrases of public show descriptions, with the source URL for each.

