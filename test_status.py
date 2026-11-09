import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from common import CATALOG, CLEAR_PITCH, MOCK_CFG, VAGUE_PITCH, good_report  # noqa: F401

from pitchcheck import llm, pipeline
from pitchcheck.llm import ModelOutputError, ProviderError, complete, parse_json
from pitchcheck.mock import mock_complete
from pitchcheck.pipeline import CardCache, check_pitch

llm.RETRY_SLEEP = (0, 0)
llm.RATE_WAIT_SCALE = 0

EXTRACT_OK = lambda text: json.dumps({"events": [  # noqa: E731
    {"role": "opening_conflict", "text": "partner leaves her", "excerpt": "leaves her for another designer"},
    {"role": "protagonist_choice", "text": "she marries a stranger", "excerpt": "immediately marries a stranger"},
]})


class Patch:
    """Route model calls by task, restoring the real function afterwards."""

    def __init__(self, extract=None, compare=None, catalog_extract_mock=True):
        self.extract, self.compare, self.cat_mock = extract, compare, catalog_extract_mock

    def __enter__(self):
        self.real = pipeline.complete

        def routed(prompt, cfg, task=None, context=None):
            if task == "extract":
                if context.get("kind") == "catalog" or self.extract is None:
                    return mock_complete("mock", task, context)
                return self.extract(context["text"]) if callable(self.extract) else self.extract
            if task == "compare":
                return self.compare(context) if callable(self.compare) else (self.compare if self.compare is not None else mock_complete("mock", task, context))
            raise AssertionError(task)

        pipeline.complete = routed
        return self

    def __exit__(self, *a):
        pipeline.complete = self.real


def run(pitch, **patch):
    with Patch(**patch):
        return check_pitch(pitch, CATALOG, MOCK_CFG, CardCache(None))


class TestUnclearPitchVersusBrokenModel(unittest.TestCase):
    """Priority 1: a clear pitch must never receive a verified 'insufficient information' because of a model failure."""

    def assert_model_failure(self, res):
        self.assertEqual(res["status"], "unverified", res)
        self.assertEqual(res["status_reason"], "extraction_unusable")
        self.assertIsNone(res["verdict"])
        self.assertIsNone(res["claimed_verdict"])
        self.assertIn("model failure, not a finding about the pitch", res["error"])
        self.assertIsNone(res["raw"]["compare_reply"])  # no comparison on a broken extraction

    def test_detailed_pitch_with_non_json_extraction(self):
        res = run(CLEAR_PITCH, extract="Sorry, I cannot do that.")
        self.assert_model_failure(res)
        self.assertEqual(res["raw"]["pitch_extraction_reply"], "Sorry, I cannot do that.")  # raw preserved

    def test_detailed_pitch_with_wrong_json_shape(self):
        for reply in ('{"events": "none"}', "[]", '{"events": null}', "{}", '"text"'):
            self.assert_model_failure(run(CLEAR_PITCH, extract=reply))

    def test_detailed_pitch_with_invented_quotations(self):
        reply = json.dumps({"events": [
            {"role": "opening_conflict", "text": "x", "excerpt": "a sentence that the pitch never contains"},
            {"role": "reveal", "text": "y", "excerpt": "another invented quotation about a secret"},
        ]})
        res = run(CLEAR_PITCH, extract=reply)
        self.assert_model_failure(res)
        self.assertEqual(len(res["pitch_extraction"]["dropped"]), 2)

    def test_detailed_pitch_with_empty_but_valid_event_list(self):
        res = run(CLEAR_PITCH, extract='{"events": []}')
        self.assert_model_failure(res)  # 34 words but zero events: the model failed, the pitch is not vague

    def test_detailed_pitch_where_most_events_were_discarded(self):
        reply = json.dumps({"events": [
            {"role": "opening_conflict", "text": "partner leaves her", "excerpt": "leaves her for another designer"},
            {"role": "reveal", "text": "y", "excerpt": "an invented quotation about a hidden twin"},
        ]})
        self.assert_model_failure(run(CLEAR_PITCH, extract=reply))

    def test_genuinely_vague_pitch_with_valid_extraction_asks_one_question(self):
        reply = json.dumps({"events": [{"role": "opening_conflict", "text": "a photographer falls for a billionaire", "excerpt": "photographer falls for a billionaire"}]})
        res = run(VAGUE_PITCH, extract=reply)
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["verdict"], "insufficient_information")
        self.assertEqual(res["status_reason"], "code_gate_thin_pitch")
        q = res["report"]["editor_question"]
        self.assertEqual(q.count("?"), 1)
        self.assertIn("decide", q)  # asks about the missing decision
        self.assertIsNone(res["raw"]["compare_reply"])
        self.assertIn("code gate", res["decided_by"])

    def test_vague_pitch_with_zero_events_and_valid_reply_is_vague_not_broken(self):
        res = run("An heiress faces a life-changing choice.", extract='{"events": []}')
        self.assertEqual((res["status"], res["verdict"]), ("verified", "insufficient_information"))

    def test_vague_pitch_whose_extraction_is_broken_is_not_a_verified_verdict(self):
        res = run(VAGUE_PITCH, extract="garbage")
        self.assertEqual(res["status"], "unverified")
        self.assertIsNone(res["verdict"])

    def test_pitch_without_a_final_reveal_is_still_compared(self):
        # two stated events (a conflict and a decision), no reveal: must reach the comparison step
        res = run(CLEAR_PITCH, extract=EXTRACT_OK)
        self.assertIsNotNone(res["raw"]["compare_reply"])
        self.assertNotEqual(res["status_reason"], "code_gate_thin_pitch")
        self.assertIn("reveal", res["pitch_extraction"]["missing_roles"])

    def test_follow_up_question_names_the_missing_piece(self):
        card = {"missing_roles": ["reveal"]}
        self.assertIn("result", pipeline.follow_up_question(card))
        self.assertIn("decide", pipeline.follow_up_question({"missing_roles": ["protagonist_choice", "reveal"]}))
        self.assertIn("start", pipeline.follow_up_question({"missing_roles": ["opening_conflict"]}))


class TestUnsupportedComparison(unittest.TestCase):
    def test_non_json_comparison_reply_is_unverified_and_preserved(self):
        res = run(CLEAR_PITCH, extract=EXTRACT_OK, compare="I think these are similar.")
        self.assertEqual((res["status"], res["status_reason"]), ("unverified", "reply_not_json"))
        self.assertEqual(res["raw"]["compare_reply"], "I think these are similar.")
        self.assertIsNone(res["verdict"])

    def test_fabricated_report_is_unverified_with_the_claim_kept_separately(self):
        res = run(CLEAR_PITCH, extract=EXTRACT_OK, compare=lambda ctx: mock_complete("mock_faulty", "compare", ctx))
        self.assertEqual(res["status"], "unverified")
        self.assertIsNone(res["verdict"])
        self.assertEqual(res["claimed_verdict"], "same_sequence")  # a guess, not a verified result
        self.assertEqual(res["checks"]["quotes_verified"], 0)
        self.assertTrue(res["raw"]["compare_reply"])

    def test_wrong_types_in_the_comparison_do_not_crash(self):
        bad = json.dumps({"verdict": "same_sequence", "closest_id": 5, "shared_events": "oops", "first_difference": [1]})
        res = run(CLEAR_PITCH, extract=EXTRACT_OK, compare=bad)
        self.assertEqual(res["status"], "unverified")

    def test_insufficient_verdict_on_a_detailed_pitch_is_not_verified(self):
        rep = json.dumps({"verdict": "insufficient_information", "closest_id": None, "shared_events": [], "first_difference": None,
                          "editor_question": "What does she decide?"})
        three = lambda t: json.dumps({"events": [  # noqa: E731
            {"role": "opening_conflict", "text": "a", "excerpt": "leaves her for another designer"},
            {"role": "protagonist_choice", "text": "b", "excerpt": "immediately marries a stranger"},
            {"role": "reveal", "text": "c", "excerpt": "main rival of her ex's firm"}]})
        res = run(CLEAR_PITCH, extract=three, compare=rep)
        self.assertEqual(res["status"], "unverified")
        self.assertEqual(res["claimed_verdict"], "insufficient_information")

    def test_a_valid_report_is_verified(self):
        res = run(CLEAR_PITCH, extract=EXTRACT_OK, compare=json.dumps(good_report()))
        self.assertEqual((res["status"], res["verdict"]), ("verified", "same_sequence"))
        self.assertEqual(res["report"]["closest_id"], "SHOW_001")


class TestCaching(unittest.TestCase):
    def test_malformed_catalog_cards_are_never_cached(self):
        tmp = os.path.join(tempfile.mkdtemp(), "cache.json")
        cache = CardCache(tmp)
        real = pipeline.complete
        pipeline.complete = lambda p, c, task=None, context=None: "garbage"
        try:
            card = cache.get("Some synopsis with several words in it.", MOCK_CFG, "catalog")
        finally:
            pipeline.complete = real
        self.assertFalse(card["well_formed"])
        self.assertFalse(os.path.exists(tmp))

    def test_good_cards_are_cached_and_reused(self):
        tmp = os.path.join(tempfile.mkdtemp(), "cache.json")
        check_pitch(CLEAR_PITCH, CATALOG, MOCK_CFG, CardCache(tmp))
        with open(tmp, encoding="utf-8") as f:
            n = len(json.load(f))
        self.assertEqual(n, len(CATALOG))
        calls = []
        real = pipeline.complete
        pipeline.complete = lambda p, c, task=None, context=None: (calls.append(task), real(p, c, task, context))[1]
        try:
            check_pitch(CLEAR_PITCH, CATALOG, MOCK_CFG, CardCache(tmp))
        finally:
            pipeline.complete = real
        self.assertEqual(calls.count("extract"), 1)  # only the pitch; the catalog came from the cache


class TestInputsAndModes(unittest.TestCase):
    def test_empty_and_oversized_pitches_are_invalid_input(self):
        for p in ("", "   ", "x" * 5000):
            res = check_pitch(p, CATALOG, MOCK_CFG, CardCache(None))
            self.assertEqual(res["status"], "invalid_input")

    def test_full_comparison_is_the_default_and_shortlist_is_opt_in(self):
        full = check_pitch(CLEAR_PITCH, CATALOG, {"provider": "mock", "model": "mock"}, CardCache(None))
        self.assertEqual(full["mode"], "full")
        self.assertEqual(len(full["candidates"]), len(CATALOG))
        short = check_pitch(CLEAR_PITCH, CATALOG, MOCK_CFG, CardCache(None), mode="shortlist", k=3)
        self.assertEqual(len(short["candidates"]), 3)

    def test_internal_defect_is_reported_separately(self):
        real = pipeline.check_report
        pipeline.check_report = lambda *a, **k: 1 / 0
        try:
            res = check_pitch(CLEAR_PITCH, CATALOG, MOCK_CFG, CardCache(None))
        finally:
            pipeline.check_report = real
        self.assertEqual(res["status"], "internal_error")
        self.assertIn("ZeroDivisionError", res["error"])


class FakeAPI(BaseHTTPRequestHandler):
    seen = []
    mode = "ok"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.seen.append((self.path, dict(self.headers), body))
        m = FakeAPI.mode
        if m == "slow":
            time.sleep(1.0)
        if m in ("401", "429", "404", "500"):
            data = json.dumps({"error": {"message": f"fake {m}"}}).encode()
            self.send_response(int(m))
        elif m == "html":
            data = b"<html>not an api</html>"
            self.send_response(200)
        elif m == "shape":
            data = json.dumps({"unexpected": True}).encode()
            self.send_response(200)
        else:
            out = ({"candidates": [{"content": {"parts": [{"text": '{"ok": "gemini"}'}]}}]} if "generateContent" in self.path
                   else {"choices": [{"message": {"content": '{"ok": "openai"}'}}]})
            data = json.dumps(out).encode()
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:
            pass

    def log_message(self, *a):
        pass


class TestProviderErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeAPI)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        FakeAPI.mode = "ok"
        os.environ["TEST_KEY"] = "secret-k-123"

    def cfg(self, prov="openai_compat", **kw):
        base = f"http://127.0.0.1:{self.port}/v1" if prov == "openai_compat" else f"http://127.0.0.1:{self.port}/v1beta"
        return dict({"provider": prov, "model": "m", "api_key_env": "TEST_KEY", "base_url": base, "timeout_s": 5}, **kw)

    def test_requests_are_built_correctly_and_the_key_stays_out_of_the_url(self):
        out = complete("hello", self.cfg("gemini"))
        self.assertEqual(json.loads(out), {"ok": "gemini"})
        path, headers, body = FakeAPI.seen[-1]
        self.assertTrue(path.endswith("/models/m:generateContent"))
        self.assertNotIn("secret-k-123", path)
        self.assertEqual({k.lower(): v for k, v in headers.items()}["x-goog-api-key"], "secret-k-123")
        out = complete("hello", self.cfg("openai_compat"))
        self.assertEqual(json.loads(out), {"ok": "openai"})
        self.assertEqual(FakeAPI.seen[-1][1].get("Authorization"), "Bearer secret-k-123")

    def test_missing_key_is_an_actionable_provider_error_not_a_mock_fallback(self):
        os.environ.pop("NO_SUCH_KEY_VAR", None)
        cfg = self.cfg(api_key_env="NO_SUCH_KEY_VAR")
        res = check_pitch(CLEAR_PITCH, CATALOG, cfg, CardCache(None))
        self.assertEqual((res["status"], res["status_reason"]), ("provider_error", "missing_key"))
        self.assertFalse(res["mock"])
        self.assertIn("$env:NO_SUCH_KEY_VAR", res["provider_error"]["hint"])
        self.assertIsNone(res["verdict"])

    def test_http_errors_are_provider_errors_with_hints(self):
        for code, word in (("401", "credentials"), ("429", "rate limit"), ("404", "Model or URL")):
            FakeAPI.mode = code
            res = check_pitch(CLEAR_PITCH, CATALOG, self.cfg(), CardCache(None))
            self.assertEqual(res["status"], "provider_error", code)
            self.assertEqual(res["status_reason"], f"http_{code}")
            self.assertIn(word.lower(), res["provider_error"]["hint"].lower())
            self.assertNotIn("secret-k-123", json.dumps(res))

    def test_timeout_is_a_provider_error(self):
        FakeAPI.mode = "slow"
        res = check_pitch(CLEAR_PITCH, CATALOG, self.cfg(timeout_s=0.2), CardCache(None))
        self.assertEqual((res["status"], res["status_reason"]), ("provider_error", "timeout"))
        self.assertIn("timeout_s", res["provider_error"]["hint"])

    def test_unreachable_host_is_a_network_error(self):
        res = check_pitch(CLEAR_PITCH, CATALOG, self.cfg(base_url="http://127.0.0.1:9/v1", timeout_s=2), CardCache(None))
        self.assertEqual(res["status"], "provider_error")
        self.assertIn(res["status_reason"], ("network", "timeout"))

    def test_odd_provider_replies_are_provider_errors(self):
        for mode in ("html", "shape"):
            FakeAPI.mode = mode
            with self.assertRaises(ProviderError) as cm:
                complete("x", self.cfg())
            self.assertEqual(cm.exception.kind, "bad_response")

    def test_unknown_provider_and_missing_base_url(self):
        with self.assertRaises(ProviderError) as cm:
            complete("x", {"provider": "nope", "model": "m", "api_key_env": "TEST_KEY"})
        self.assertEqual(cm.exception.kind, "config")
        with self.assertRaises(ProviderError):
            complete("x", {"provider": "openai_compat", "model": "m", "api_key_env": "TEST_KEY"})

    def test_parse_json(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ModelOutputError):
            parse_json("nope")


if __name__ == "__main__":
    unittest.main()


class TestPromptConsistency(unittest.TestCase):
    """The prompt must not offer a shape that the checker then rejects (the v3 live run: first_difference null on a match)."""

    def test_prompt_requires_first_difference_object_for_match_verdicts(self):
        from pitchcheck import prompts
        txt = prompts.COMPARE
        self.assertNotIn('"catalog_excerpt":"... or null"} or null', txt)
        self.assertIn("REQUIRED as an object (never null)", txt)


class TestComparePromptGivesOnlyVerbatimQuotes(unittest.TestCase):
    def test_event_lists_carry_quotes_not_paraphrases(self):
        from pitchcheck.pipeline import build_compare_prompt
        pitch_card = {"events": [{"role": "reveal", "text": "SUMMARY-PITCH-WORDS", "excerpt": "she marries a stranger at once"}]}
        cards = {r["id"]: {"events": [{"role": "reveal", "text": "SUMMARY-CATALOG-WORDS", "excerpt": "Elle quickly marries Roy"}]} for r in CATALOG}
        prompt = build_compare_prompt("she marries a stranger at once", pitch_card, CATALOG, cards)
        self.assertIn("she marries a stranger at once", prompt)
        self.assertIn("Elle quickly marries Roy", prompt)
        self.assertNotIn("SUMMARY-PITCH-WORDS", prompt)
        self.assertNotIn("SUMMARY-CATALOG-WORDS", prompt)


class TestLiveRunRegressions(unittest.TestCase):
    """Defects seen in the first live dev run (Groq / gpt-oss-120b)."""

    def test_non_breaking_hyphen_in_a_quote_still_matches(self):
        from pitchcheck.textutil import find_range
        self.assertIsNotNone(find_range("An heiress faces a life‑changing choice", "An heiress faces a life-changing choice."))

    def test_comparison_step_sees_only_its_own_kind_words(self):
        from pitchcheck.pipeline import build_compare_prompt
        pc = {"events": [{"role": "other", "text": "t", "excerpt": "her family accuses him of chasing her fortune"}]}
        cards = {r["id"]: {"events": [{"role": "protagonist_choice", "text": "t", "excerpt": "Elle quickly marries Roy"}]} for r in CATALOG}
        prompt = build_compare_prompt("x", pc, CATALOG, cards)
        self.assertIn('"kind": "consequence"', prompt)
        self.assertIn('"kind": "decision"', prompt)
        self.assertNotIn('"role": "other"', prompt)
