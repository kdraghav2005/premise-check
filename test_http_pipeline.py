"""End to end through the real openai_compat adapter, over HTTP, against a LOCAL stand-in server.
This proves the request/response plumbing of the live path. The replies are canned: it says nothing about a real model."""
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from common import CATALOG, CLEAR_PITCH, good_report  # noqa: F401

import eval as ev  # noqa: F401  (ensures eval imports with the same sys.path)
from pitchcheck import llm
from pitchcheck.pipeline import CardCache, check_pitch
from pitchcheck.textutil import sentences

llm.RETRY_SLEEP = (0, 0)
llm.RATE_WAIT_SCALE = 0


class StandIn(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        if "list its story events" in prompt:
            text = prompt.split("TEXT:\n", 1)[1]
            reply = {"events": [{"role": "opening_conflict" if i == 0 else "other", "text": s, "excerpt": s} for i, s in enumerate(sentences(text)[:5])]}
            if text.startswith("After her fiance leaves her"):  # the pitch: give it labelled events
                reply = {"events": [
                    {"role": "opening_conflict", "text": "her partner leaves her", "excerpt": "leaves her for another designer"},
                    {"role": "protagonist_choice", "text": "she marries a stranger", "excerpt": "immediately marries a stranger"},
                    {"role": "reveal", "text": "he is her ex's rival", "excerpt": "main rival of her ex's firm"}]}
        else:
            reply = good_report()
        StandIn.calls.append("extract" if "list its story events" in prompt else "compare")
        data = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class TestLivePathOverHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), StandIn)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_full_flow_through_the_openai_compatible_adapter(self):
        os.environ["HTTP_TEST_KEY"] = "k"
        cfg = {"provider": "openai_compat", "model": "stand-in", "api_key_env": "HTTP_TEST_KEY",
               "base_url": f"http://127.0.0.1:{self.port}/v1", "timeout_s": 5, "retrieval": {"mode": "full"}}
        StandIn.calls.clear()
        res = check_pitch(CLEAR_PITCH, CATALOG, cfg, CardCache(os.path.join(tempfile.mkdtemp(), "c.json")))
        self.assertEqual((res["status"], res["verdict"]), ("verified", "same_sequence"), res.get("error"))
        self.assertFalse(res["mock"])
        self.assertEqual(res["report"]["closest_id"], "SHOW_001")
        self.assertEqual(StandIn.calls.count("extract"), len(CATALOG) + 1)  # 10 catalog shows (once) + the pitch
        self.assertEqual(StandIn.calls.count("compare"), 1)  # one comparison against ALL shows, no shortlist
        self.assertEqual(len(res["candidates"]), len(CATALOG))


if __name__ == "__main__":
    unittest.main()


class UAHandler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        UAHandler.seen.append(self.headers.get("User-Agent", ""))
        if "urllib" in self.headers.get("User-Agent", "").lower():  # what Cloudflare/Groq does: 403, error code 1010
            data, code = b"error code: 1010", 403
        else:
            data, code = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode(), 200
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class TestUserAgent(unittest.TestCase):
    def test_requests_do_not_use_the_default_python_urllib_agent(self):
        srv = HTTPServer(("127.0.0.1", 0), UAHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            os.environ["UA_TEST_KEY"] = "k"
            cfg = {"provider": "openai_compat", "model": "m", "api_key_env": "UA_TEST_KEY",
                   "base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1", "timeout_s": 5}
            self.assertEqual(llm.complete("hi", cfg), "{}")
            self.assertNotIn("urllib", UAHandler.seen[-1].lower())
        finally:
            srv.shutdown()
            srv.server_close()


class RateLimitHandler(BaseHTTPRequestHandler):
    fail_first = 2
    count = 0
    waits = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        RateLimitHandler.count += 1
        if RateLimitHandler.count <= RateLimitHandler.fail_first:
            data, code = b'{"error":{"message":"Rate limit reached ... Please try again in 6.69s. Need more tokens?"}}', 429
        else:
            data, code = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode(), 200
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class TestRateLimit(unittest.TestCase):
    def _cfg(self, srv):
        os.environ["RL_TEST_KEY"] = "k"
        return {"provider": "openai_compat", "model": "m", "api_key_env": "RL_TEST_KEY",
                "base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1", "timeout_s": 5}

    def _serve(self, fail_first):
        RateLimitHandler.fail_first, RateLimitHandler.count = fail_first, 0
        srv = HTTPServer(("127.0.0.1", 0), RateLimitHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_recovers_after_more_than_two_rate_limit_replies(self):
        srv = self._serve(fail_first=5)  # the old fixed 2s+4s retry gave up after 3 tries
        try:
            self.assertEqual(llm.complete("hi", self._cfg(srv)), "{}")
            self.assertEqual(RateLimitHandler.count, 6)
        finally:
            srv.shutdown(); srv.server_close()

    def test_gives_up_with_http_429_after_the_limit(self):
        srv = self._serve(fail_first=999)
        try:
            with self.assertRaises(llm.ProviderError) as cm:
                llm.complete("hi", self._cfg(srv))
            self.assertEqual(cm.exception.kind, "http_429")
            self.assertEqual(RateLimitHandler.count, llm.MAX_RATE_RETRIES + 1)
        finally:
            srv.shutdown(); srv.server_close()

    def test_wait_follows_the_providers_hint(self):
        old = llm.RATE_WAIT_SCALE
        llm.RATE_WAIT_SCALE = 1.0
        try:
            self.assertAlmostEqual(llm._rate_wait_seconds("try again in 6.69s.", None, 1), 7.69, places=2)
            self.assertAlmostEqual(llm._rate_wait_seconds("try again in 1m2.5s.", None, 1), 63.5, places=2)
            self.assertAlmostEqual(llm._rate_wait_seconds("", "3", 1), 4.0, places=2)
            self.assertEqual(llm._rate_wait_seconds("try again in 9m0s", None, 1), llm.MAX_RATE_WAIT)
        finally:
            llm.RATE_WAIT_SCALE = old
