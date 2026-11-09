import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.parse

from common import CATALOG, CLEAR_PITCH, MOCK_CFG, VAGUE_PITCH  # noqa: F401

import server
from pitchcheck import llm, pipeline
from pitchcheck.mock import mock_complete
from pitchcheck.render import render_page, render_result, safe_url

llm.RETRY_SLEEP = (0, 0)
llm.RATE_WAIT_SCALE = 0


class Base(unittest.TestCase):
    cfg = MOCK_CFG
    catalog = CATALOG

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.srv = server.make_server("127.0.0.1", 0, dict(cls.cfg), cls.catalog, os.path.join(cls.tmp, "runs"), os.path.join(cls.tmp, "cache.json"))
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read().decode("utf-8")
        out = (r.status, dict(r.getheaders()), data)
        c.close()
        return out

    def submit(self, pitch):
        body = urllib.parse.urlencode({"pitch": pitch})
        status, headers, _ = self.req("POST", "/check", body, {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(status, 303)
        loc = headers["Location"]
        status, _, page = self.req("GET", loc)
        self.assertEqual(status, 200)
        return loc, page


class TestMockScreen(Base):
    def test_home_has_pitch_box_compare_button_and_unmistakable_mock_banner(self):
        _, _, page = self.req("GET", "/")
        self.assertIn("<textarea", page)
        self.assertIn(">Compare</button>", page)
        self.assertIn("MOCK PROVIDER", page)

    def test_report_is_structured_not_a_text_dump(self):
        _, page = self.submit("Her fiance calls off their engagement in front of his whole firm. She agrees to a paper marriage with a rich widower, finds out she is the long-lost daughter of a shipping family, and takes over the company. The paper marriage turns into a real one.")
        for needle in ("Closest show", "Shared events, in order", "Evidence from catalog synopsis", "From your pitch", "Technical details",
                       "Download this run", "First difference: not established by the sources", "Question for the editor", "MOCK PROVIDER"):
            self.assertIn(needle, page)
        self.assertIn('href="https://www.reelshort.com/', page)  # clickable source
        self.assertIn('rel="noopener noreferrer"', page)
        self.assertNotIn("<pre>VERDICT", page)
        # raw JSON sits inside a closed disclosure
        self.assertRegex(page, r"<details><summary>Technical details</summary>")

    def test_vague_pitch_shows_one_question_and_no_match_claims(self):
        _, page = self.submit(VAGUE_PITCH)
        self.assertIn("Insufficient information", page)
        self.assertIn("Question to answer first", page)
        self.assertNotIn("Closest show", page)
        self.assertNotIn("Shared events", page)

    def test_download_contains_the_whole_run_and_no_secrets(self):
        os.environ["PITCHCHECK_API_KEY"] = "do-not-leak-me"
        loc, _ = self.submit(CLEAR_PITCH)
        run_id = loc.rsplit("/", 1)[1]
        status, headers, body = self.req("GET", f"/download/{run_id}.json")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        run = json.loads(body)
        for key in ("pitch", "provider", "model", "status", "status_reason", "checks", "raw", "report", "run_id", "config"):
            self.assertIn(key, run)
        self.assertEqual(run["pitch"], CLEAR_PITCH)
        self.assertIn("pitch_extraction_reply", run["raw"])
        self.assertNotIn("do-not-leak-me", body)

    def test_bad_requests(self):
        self.assertEqual(self.req("GET", "/run/../../etc/passwd")[0], 404)
        self.assertEqual(self.req("GET", "/run/ffffffffffff")[0], 404)
        self.assertEqual(self.req("GET", "/download/%2e%2e%2fsecret.json")[0], 404)
        self.assertEqual(self.req("POST", "/check", "x" * 30000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.req("GET", "/nope")[0], 404)

    def test_empty_pitch_is_shown_as_invalid_input_and_saved(self):
        _, page = self.submit("   ")
        self.assertIn("INVALID INPUT", page)

    def test_pitch_text_is_escaped(self):
        _, page = self.submit('<script>alert("x")</script> She leaves him. He follows her to the city.')
        self.assertNotIn('<script>alert("x")</script>', page)
        self.assertIn("&lt;script&gt;", page)


class TestRealProviderWithoutKey(Base):
    cfg = {"provider": "gemini", "model": "m", "api_key_env": "NO_KEY_IN_TEST_ENV", "retrieval": {"mode": "full"}}

    def test_missing_key_is_a_provider_error_page_not_a_mock(self):
        os.environ.pop("NO_KEY_IN_TEST_ENV", None)
        loc, page = self.submit(CLEAR_PITCH)
        self.assertIn("PROVIDER ERROR", page)
        self.assertIn("No comparison was made", page)
        self.assertIn("$env:NO_KEY_IN_TEST_ENV", page)
        self.assertNotIn("MOCK PROVIDER", page)
        self.assertNotIn("Meaningful sequence overlap", page)
        status, _, body = self.req("GET", f"/download/{loc.rsplit('/', 1)[1]}.json")
        self.assertEqual(json.loads(body)["status"], "provider_error")  # failed runs stay available


class TestUnverifiedScreen(Base):
    def test_unverified_run_is_loud_and_keeps_the_raw_reply(self):
        real = pipeline.complete

        def routed(prompt, cfg, task=None, context=None):
            if task == "extract" and context.get("kind") == "pitch":
                return "I'm sorry, I can't produce JSON."
            return mock_complete("mock", task, context)

        pipeline.complete = routed
        try:
            loc, page = self.submit(CLEAR_PITCH)
        finally:
            pipeline.complete = real
        self.assertIn("UNVERIFIED", page)
        self.assertIn("Do not rely on this result", page)
        self.assertIn("model failure, not a finding about the pitch", page)
        self.assertNotIn("Insufficient information", page)
        self.assertIn("I&#x27;m sorry, I can&#x27;t produce JSON.", page)  # raw reply is in the disclosure
        run = json.loads(self.req("GET", f"/download/{loc.rsplit('/', 1)[1]}.json")[2])
        self.assertEqual(run["raw"]["pitch_extraction_reply"], "I'm sorry, I can't produce JSON.")


class TestRendererIsCrashProof(unittest.TestCase):
    def test_wrong_types_everywhere(self):
        weird = {
            "run_id": "abc123abc123", "status": "unverified", "status_reason": 5, "mock": True, "claimed_verdict": ["x"],
            "report": {"verdict": 5, "closest_id": ["SHOW_001"], "shared_events": "oops",
                       "first_difference": [1, 2], "editor_question": {"a": 1}, "missing_info": "x"},
            "checks": {"items": "no", "failures": 3, "warnings": None}, "error": {"x": 1}, "provider_error": "text",
        }
        html = render_result(weird, CATALOG)
        self.assertIn("UNVERIFIED", html)
        weird2 = dict(weird, status="verified", verdict="same_sequence",
                      report={"closest_id": "SHOW_001", "shared_events": [None, 3, {"pitch_excerpt": 5, "kind": []}],
                              "first_difference": {"established": True, "explanation": 7}})
        self.assertIn("Shared events", render_result(weird2, CATALOG))
        for st in ("provider_error", "internal_error", "invalid_input", "verified", "unverified"):
            render_result({"run_id": "abc123abc123", "status": st, "report": None}, CATALOG)

    def test_catalog_fields_are_escaped_and_bad_urls_are_not_linked(self):
        evil = [{"id": "X", "title": "<img src=x onerror=alert(1)>", "synopsis": "s", "source_url": "javascript:alert(1)"}]
        res = {"run_id": "abc123abc123", "status": "verified", "verdict": "shared_trope_only",
               "report": {"verdict": "shared_trope_only", "closest_id": "X", "shared_events": []}}
        html = render_result(res, evil)
        self.assertNotIn("<img src=x", html)
        self.assertNotIn('href="javascript', html)
        self.assertIsNone(safe_url("javascript:alert(1)"))
        self.assertIsNone(safe_url("https://a.b/x y"))
        self.assertEqual(safe_url("https://a.b/x"), "https://a.b/x")

    def test_real_provider_page_has_no_mock_banner(self):
        self.assertNotIn("MOCK PROVIDER", render_page({"provider": "gemini", "model": "m"}, CATALOG))
        self.assertIn("MOCK PROVIDER", render_page({"provider": "mock", "model": "mock"}, CATALOG))


if __name__ == "__main__":
    unittest.main()
