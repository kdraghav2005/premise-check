"""v7: near-miss quote correction (disclosed) and the single feedback retry. Canned replies; says nothing about a real model."""
import copy
import json
import unittest
from unittest import mock

from common import BY_ID, CATALOG, CLEAR_PITCH, MOCK_CFG, good_report

from pitchcheck import pipeline
from pitchcheck.llm import ProviderError, mock_complete
from pitchcheck.pipeline import CardCache, check_pitch, repair_quotes
from pitchcheck.textutil import closest_span

SYN8 = BY_ID["SHOW_008"]["synopsis"]


class TestClosestSpan(unittest.TestCase):
    def test_live_run_t02_dropped_word(self):
        # live run, case T02: the model wrote "patient turns out be their daughter." for "turns out to be"
        self.assertEqual(closest_span("patient turns out be their daughter.", SYN8), "patient turns out to be their daughter.")

    def test_invented_sentence_is_not_matched(self):
        # live run, case T04: the quoted synopsis text does not exist in the synopsis at all
        self.assertIsNone(closest_span("discovers he is her ex’s business rival", BY_ID["SHOW_002"]["synopsis"]))

    def test_too_many_words_off(self):
        self.assertIsNone(closest_span("a doctor refuses to treat their daughter at once", SYN8))

    def test_short_quotes_need_to_be_exact(self):
        self.assertIsNone(closest_span("quickly marry Roy", BY_ID["SHOW_001"]["synopsis"]))


class TestRepairQuotes(unittest.TestCase):
    def test_near_miss_is_corrected_and_listed(self):
        rep = good_report()
        rep["shared_events"][0]["catalog_excerpt"] = "husband leaves her a woman impersonating"  # 'for' dropped
        fixed, repairs = repair_quotes(rep, CLEAR_PITCH, BY_ID)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(fixed["shared_events"][0]["catalog_excerpt"], "husband leaves her for a woman impersonating")
        self.assertEqual(rep["shared_events"][0]["catalog_excerpt"], "husband leaves her a woman impersonating")  # input untouched

    def test_exact_quotes_are_left_alone(self):
        fixed, repairs = repair_quotes(good_report(), CLEAR_PITCH, BY_ID)
        self.assertEqual(repairs, [])
        self.assertEqual(fixed, good_report())

    def test_invented_quote_stays_and_still_fails(self):
        rep = good_report()
        rep["first_difference"]["catalog_excerpt"] = "she opens a bakery in another city entirely"
        fixed, repairs = repair_quotes(rep, CLEAR_PITCH, BY_ID)
        self.assertEqual(repairs, [])
        _, checks, _ = pipeline._verify(rep, CLEAR_PITCH, CATALOG, {"events": 3, "words": 40})
        self.assertFalse(checks["ok"])


def _run(compare_replies, retry_error=False):
    calls = []

    def fake(prompt, cfg, task=None, context=None):
        if task == "extract":
            return mock_complete("mock", task, context)
        calls.append(prompt)
        if len(calls) > 1 and retry_error:
            raise ProviderError("timeout", "timed out", "retry later")
        return json.dumps(compare_replies[min(len(calls), len(compare_replies)) - 1])

    with mock.patch.object(pipeline, "complete", fake):
        res = check_pitch(CLEAR_PITCH, CATALOG, MOCK_CFG, CardCache(None))
    return res, calls


def _bad():
    rep = good_report()
    rep["first_difference"]["catalog_excerpt"] = "she opens a bakery in another city entirely"
    return rep


class TestRetry(unittest.TestCase):
    def test_first_reply_ok_means_no_retry(self):
        res, calls = _run([good_report()])
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["status_reason"], "checks_passed")
        self.assertNotIn("retry", res)

    def test_near_miss_is_fixed_without_a_retry(self):
        rep = good_report()
        rep["shared_events"][0]["catalog_excerpt"] = "husband leaves her a woman impersonating"
        res, calls = _run([rep])
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["status"], "verified")
        self.assertEqual(len(res["quote_repairs"]), 1)
        self.assertTrue(any("Quote corrected" in w for w in res["checks"]["warnings"]))

    def test_retry_that_passes_is_verified_and_labelled(self):
        res, calls = _run([_bad(), good_report()])
        self.assertEqual(len(calls), 2)
        self.assertIn("failed these mechanical checks", calls[1])
        self.assertIn("bakery", calls[1])  # the model sees its own bad reply
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["status_reason"], "checks_passed_after_retry")
        self.assertEqual(res["retry"]["outcome"], "retry passed the checks")
        self.assertTrue(any("second attempt" in w for w in res["checks"]["warnings"]))
        self.assertIn("bakery", res["raw"]["compare_reply"])
        self.assertIsNotNone(res["raw"]["retry_reply"])

    def test_retry_that_fails_stays_unverified_and_keeps_first_reply(self):
        res, calls = _run([_bad(), _bad()])
        self.assertEqual(len(calls), 2)  # one retry, never more
        self.assertEqual(res["status"], "unverified")
        self.assertIsNone(res["verdict"])
        self.assertEqual(res["claimed_verdict"], "same_sequence")
        self.assertTrue(res["retry"]["outcome"].startswith("retry still failed"))

    def test_provider_error_in_retry_keeps_first_result(self):
        res, calls = _run([_bad()], retry_error=True)
        self.assertEqual(res["status"], "unverified")
        self.assertEqual(res["status_reason"], "report_failed_checks")
        self.assertIn("timeout", res["retry"]["outcome"])
        self.assertIsNotNone(res["report"])


if __name__ == "__main__":
    unittest.main()


class TestGenerationFailureRetry(unittest.TestCase):
    """Live run (Groq, gpt-oss-120b): HTTP 400 json_validate_failed / 'max completion tokens reached'. Local stand-in server."""

    def _serve(self, fail_times):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        state = {"n": 0}
        bad = json.dumps({"error": {"message": "Failed to generate JSON.", "type": "invalid_request_error", "code": "json_validate_failed",
                                    "failed_generation": "max completion tokens reached before generating a valid document"}}).encode()
        ok = json.dumps({"choices": [{"message": {"content": "{\"a\": 1}"}}]}).encode()

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                state["n"] += 1
                body, code = (bad, 400) if state["n"] <= fail_times else (ok, 200)
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return srv, state

    def _cfg(self, srv):
        return {"provider": "openai_compat", "model": "m", "base_url": f"http://127.0.0.1:{srv.server_port}/v1", "api_key_env": "T7_KEY", "timeout_s": 5}

    def test_recovers_after_two_failures(self):
        import os
        from pitchcheck import llm
        srv, state = self._serve(2)
        with mock.patch.dict(os.environ, {"T7_KEY": "k"}), mock.patch.object(llm, "RETRY_SLEEP", (0, 0)):
            out = llm.complete("p", self._cfg(srv))
        self.assertEqual(out, "{\"a\": 1}")
        self.assertEqual(state["n"], 3)

    def test_gives_up_after_the_limit_with_a_clear_hint(self):
        import os
        from pitchcheck import llm
        srv, state = self._serve(99)
        with mock.patch.dict(os.environ, {"T7_KEY": "k"}), mock.patch.object(llm, "RETRY_SLEEP", (0, 0)):
            with self.assertRaises(ProviderError) as cm:
                llm.complete("p", self._cfg(srv))
        self.assertEqual(state["n"], 3)  # first try + 2 retries, never more
        self.assertIn("ran out of output budget", cm.exception.hint)
        self.assertEqual(cm.exception.kind, "http_400")

    def test_other_400_is_not_retried(self):
        from pitchcheck import llm
        # a 400 without the generation-failure markers must fail immediately
        self.assertFalse(llm._is_generation_failure(400, '{"error":{"message":"model not found"}}'))
        self.assertTrue(llm._is_generation_failure(400, "max completion tokens reached"))
        self.assertFalse(llm._is_generation_failure(500, "json_validate_failed"))
