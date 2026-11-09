"""The naive baseline's grading and runner. Canned replies; says nothing about a real model."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from common import BY_ID, CATALOG

import naive_baseline as nb
from pitchcheck import llm

llm.RATE_WAIT_SCALE = 0

CASE = {"test_id": "X1", "split": "dev", "pitch": "A paramedic is taking the last dose of antivenom to an injured climber. A radio call reveals the climber is their son.",
        "expected_outcome": "sequence_overlap", "reference_show_id": "SHOW_008"}


def reply(**kw):
    base = {"verdict": "same_sequence", "closest_id": "SHOW_008", "reason": "r",
            "evidence": [{"pitch_quote": "A radio call reveals the climber is their son.", "synopsis_quote": "The patient turns out to be their daughter."}]}
    base.update(kw)
    return json.dumps(base)


class TestGrade(unittest.TestCase):
    def test_right_label_real_quotes(self):
        g = nb.grade_reply(reply(), CASE, BY_ID)
        self.assertTrue(g["label_ok"] and g["intended_show_named"])
        self.assertEqual((g["quotes_found"], g["quotes_checked"]), (2, 2))
        self.assertEqual(g["quotes_missing"], [])

    def test_right_label_invented_quote_is_counted(self):
        r = reply(evidence=[{"pitch_quote": "A radio call reveals the climber is their son.", "synopsis_quote": "A doctor refuses to treat the couple's own daughter at night."}])
        g = nb.grade_reply(r, CASE, BY_ID)
        self.assertTrue(g["label_ok"])
        self.assertEqual((g["quotes_found"], g["quotes_checked"]), (1, 2))
        self.assertEqual(len(g["quotes_missing"]), 1)

    def test_invalid_closest_id_makes_synopsis_quotes_unfound(self):
        g = nb.grade_reply(reply(closest_id="SHOW_999"), CASE, BY_ID)
        self.assertIsNone(g["closest_id"])
        self.assertEqual(g["quotes_found"], 1)
        self.assertFalse(g["intended_show_named"])

    def test_not_json(self):
        g = nb.grade_reply("sorry, I cannot", CASE, BY_ID)
        self.assertFalse(g["parsed"])
        self.assertFalse(g["label_ok"])

    def test_match_without_evidence(self):
        g = nb.grade_reply(reply(evidence=[]), CASE, BY_ID)
        self.assertTrue(g["match_without_evidence"])

    def test_counts_separates_right_label_from_clean_evidence(self):
        good = dict(nb.grade_reply(reply(), CASE, BY_ID), split="dev", provider_error=False)
        bad = dict(nb.grade_reply(reply(evidence=[{"pitch_quote": "made up words that are not here", "synopsis_quote": "The patient turns out to be their daughter."}]), CASE, BY_ID), split="dev", provider_error=False)
        c = nb.counts([good, bad])
        self.assertEqual(c["label agreement"], "2/2")
        self.assertEqual(c["label right AND every quote found in the source text (what the verifier would accept)"], "1/2")
        self.assertEqual(c["label right but at least one quote NOT in the source text"], "1/2")


class TestRunner(unittest.TestCase):
    def test_refuses_mock(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "mock.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({"provider": "mock", "model": "mock"}, f)
            with mock.patch("sys.argv", ["naive_baseline.py", "--config", cfg, "--out-root", d]), self.assertRaises(SystemExit) as cm:
                nb.main()
        self.assertIn("real provider", str(cm.exception))

    def test_prompt_contains_pitch_and_every_show(self):
        p = nb.build_prompt("MY PITCH TEXT", CATALOG)
        self.assertIn("MY PITCH TEXT", p)
        for r in CATALOG:
            self.assertIn(f"[id: {r['id']}]", p)
            self.assertIn(r["synopsis"], p)

    def test_runner_end_to_end_with_canned_replies(self):
        calls = []

        def fake(prompt, cfg, task=None, context=None):
            calls.append(task)
            return reply(verdict="no_meaningful_match", closest_id=None, evidence=[])

        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "live.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({"provider": "openai_compat", "model": "m", "base_url": "http://127.0.0.1:1/v1"}, f)
            with mock.patch.object(nb, "complete", fake), mock.patch("sys.argv", ["naive_baseline.py", "--config", cfg, "--out-root", d]):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = nb.main()
                self.assertEqual(rc, 0)
            runs = [x for x in os.listdir(d) if "NAIVE" in x]
            self.assertEqual(len(runs), 1)
            summ = json.load(open(os.path.join(d, runs[0], "summary.json"), encoding="utf-8"))
            self.assertEqual(summ["all"]["cases"], 12)
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(os.listdir(os.path.join(d, runs[0], "cases"))), 12)


if __name__ == "__main__":
    unittest.main()
