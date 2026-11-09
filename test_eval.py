import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from common import CATALOG, HERE, MOCK_CFG  # noqa: F401

import eval as ev
from pitchcheck import llm, pipeline, prompts
from pitchcheck.mock import mock_complete

llm.RETRY_SLEEP = (0, 0)
llm.RATE_WAIT_SCALE = 0
CASES = ev.load_cases(os.path.join(HERE, "data", "test_cases.csv"))


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def case(tid):
    return next(c for c in CASES if c["test_id"] == tid)


def res(status="verified", verdict=None, claimed=None, cid=None, reason=None):
    r = {"status": status, "status_reason": reason, "verdict": verdict, "claimed_verdict": claimed or verdict,
         "report": {"verdict": claimed or verdict, "closest_id": cid} if (claimed or verdict) else None}
    return r


def run_main(args, extra_env=None):
    """Run eval.main() in-process with a temp project state. Returns (exit_code, stdout)."""
    out = io.StringIO()
    code = 0
    with mock.patch.object(sys, "argv", ["eval.py"] + args), contextlib.redirect_stdout(out):
        try:
            code = ev.main()
        except SystemExit as e:
            code = e.code
            if isinstance(code, str):
                out.write(code)
                code = 1
    return code, out.getvalue()


class TestData(unittest.TestCase):
    def test_labels_and_splits_are_the_authored_ones(self):
        self.assertEqual(len(CASES), 12)
        self.assertEqual(sorted(c["test_id"] for c in CASES if c["split"] == "dev"), ["T01", "T02", "T04", "T05", "T07", "T09", "T11"])
        self.assertEqual(sorted(c["test_id"] for c in CASES if c["split"] == "holdout"), ["T03", "T06", "T08", "T10", "T12"])
        self.assertEqual(len(CATALOG), 10)

    def test_references_exist_and_no_pitch_is_in_the_catalog(self):
        ids = {r["id"] for r in CATALOG}
        text = " ".join(r["synopsis"] for r in CATALOG).lower()
        for c in CASES:
            if c["reference_show_id"]:
                self.assertIn(c["reference_show_id"], ids)
            self.assertNotIn(c["pitch"].lower(), text)


class TestGrading(unittest.TestCase):
    def test_unverified_answer_with_the_right_label_is_not_a_verified_success(self):
        g = ev.grade(case("T01"), res("unverified", claimed="same_sequence", cid="SHOW_001", reason="report_failed_checks"))
        self.assertTrue(g["label_ok"])
        self.assertFalse(g["verified_success"])

    def test_verified_sequence_needs_the_intended_show(self):
        self.assertTrue(ev.grade(case("T01"), res("verified", "same_sequence", cid="SHOW_001"))["verified_success"])
        g = ev.grade(case("T01"), res("verified", "same_sequence", cid="SHOW_002"))
        self.assertTrue(g["label_ok"])
        self.assertFalse(g["verified_success"])

    def test_trope_false_alarm_counts_even_when_unverified(self):
        g = ev.grade(case("T05"), res("unverified", claimed="same_sequence", cid="SHOW_003"))
        self.assertTrue(g["false_alarm"])
        g = ev.grade(case("T05"), res("verified", "shared_trope_only", cid="SHOW_002"))
        self.assertTrue(g["verified_success"])
        self.assertIn("different reference", g["reference_ok"])

    def test_provider_error_is_never_agreement(self):
        g = ev.grade(case("T09"), {"status": "provider_error", "status_reason": "timeout", "verdict": None, "claimed_verdict": None, "report": None})
        self.assertFalse(g["label_ok"])
        self.assertFalse(g["verified_success"])
        self.assertIsNone(g["claimed_verdict"])

    def test_underspecified_and_unrelated(self):
        self.assertTrue(ev.grade(case("T09"), res("verified", "insufficient_information"))["verified_success"])
        self.assertFalse(ev.grade(case("T09"), res("verified", "no_meaningful_match"))["label_ok"])
        self.assertTrue(ev.grade(case("T07"), res("verified", "no_meaningful_match"))["verified_success"])

    def test_counts_keep_the_categories_apart(self):
        rows = [
            {"exp": "same_sequence", "claimed": "same_sequence", "status": "verified", "label_ok": True, "verified_success": True, "reference_ok": True, "false_alarm": False, "qv": 6, "qc": 6, "base_rank": 1},
            {"exp": "same_sequence", "claimed": "same_sequence", "status": "unverified", "label_ok": True, "verified_success": False, "reference_ok": True, "false_alarm": False, "qv": 2, "qc": 6, "base_rank": 7},
            {"exp": "same_sequence", "claimed": None, "status": "provider_error", "label_ok": False, "verified_success": False, "reference_ok": False, "false_alarm": False, "qv": 0, "qc": 0, "base_rank": 2},
            {"exp": "shared_trope_only", "claimed": "same_sequence", "status": "unverified", "label_ok": False, "verified_success": False, "reference_ok": "n/a", "false_alarm": True, "qv": 0, "qc": 4, "base_rank": None},
        ]
        c = ev.counts(rows)
        self.assertEqual(c["label agreement (any status)"], "2/4")
        self.assertEqual(c["verified success (label right AND source checks passed)"], "1/4")
        self.assertEqual(c["label right but NOT verified"], "1/4")
        self.assertEqual(c["unverified / unsupported reports"], "2/4")
        self.assertEqual(c["provider errors (no comparison made)"], "1/4")
        self.assertEqual(c["trope_only: false alarms (claimed same_sequence)"], "1/1")
        self.assertEqual(c["KEYWORD BASELINE (retrieval only; no verdicts): intended show ranked 1st"], "1/3")
        self.assertEqual(c["KEYWORD BASELINE: intended show in top 3"], "2/3")
        self.assertEqual(c["KEYWORD BASELINE: intended show in top 5"], "2/3")
        self.assertFalse(any("baseline" in k.lower() and "verdict" in k.lower() and "correct" in k.lower() for k in c))


class TestRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [
            mock.patch.object(ev, "FREEZE_DIR", os.path.join(self.tmp, "freeze")),
            mock.patch.object(ev, "FREEZE_FILE", os.path.join(self.tmp, "freeze", "frozen.json")),
            mock.patch.object(ev, "HOLDOUT_LOG", os.path.join(self.tmp, "freeze", "holdout_runs.jsonl")),
        ]
        for p in self.patches:
            p.start()
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, *a):
        return list(a) + ["--out-root", self.out]

    def test_dev_run_with_mock_is_labelled_and_complete(self):
        code, text = run_main(self.args("--label", "d"))
        self.assertEqual(code, 0, text)
        d = os.path.join(self.out, "d")
        for f in ("summary.md", "summary.json", "grading_sheet.csv"):
            self.assertTrue(os.path.isfile(os.path.join(d, f)))
        self.assertEqual(len(os.listdir(os.path.join(d, "cases"))), 7)
        md = read(os.path.join(d, "summary.md"))
        self.assertIn("MOCK PROVIDER", md)
        self.assertIn("not frozen (dev run)", md)
        self.assertNotIn("holdout cases", md)

    def test_the_model_never_receives_the_answer_key(self):
        seen = []
        real = pipeline.complete

        def spy(prompt, cfg, task=None, context=None):
            seen.append(prompt)
            return real(prompt, cfg, task, context)

        # the verdict names are legitimately in the prompt (they are the output labels), so look for the answer-key
        # fields themselves, the authored grading notes, test ids and the case types
        keys = [c["grading_note"] for c in CASES] + [c["test_id"] for c in CASES] + [
            "expected_outcome", "grading_note", "reference_show_id", "sequence_overlap", "reskin", "trope_only", "partial_overlap", "holdout"]
        with mock.patch.object(pipeline, "complete", spy):
            code, text = run_main(self.args("--label", "k"))
        self.assertEqual(code, 0, text)
        self.assertGreater(len(seen), 10)
        import re
        for prompt in seen:
            for k in keys:
                # whole-token match: "trope_only" must not be found inside the legitimate label "shared_trope_only"
                hit = re.search(r"(?<![A-Za-z_0-9])" + re.escape(k) + r"(?![A-Za-z_0-9])", prompt)
                self.assertIsNone(hit, f"answer-key text reached the model: {k[:60]!r}")
        # eval.py itself passes only the pitch to the app
        src = read(os.path.join(HERE, "eval.py"))
        self.assertIn('check_pitch(c["pitch"], catalog, cfg, cache, mode=a.mode)', src)

    def test_holdout_needs_a_freeze(self):
        code, text = run_main(self.args("--split", "holdout"))
        self.assertNotEqual(code, 0)
        self.assertIn("No freeze found", text)

    def test_freeze_then_holdout_runs_once(self):
        self.assertEqual(run_main(self.args("--freeze"))[0], 0)
        frozen = jload(ev.FREEZE_FILE)
        self.assertEqual(frozen["prompt_version"], prompts.PROMPT_VERSION)
        for k in ("prompt_hash", "code_hash", "catalog_hash", "test_cases_hash", "config", "freeze_id"):
            self.assertIn(k, frozen)
        code, text = run_main(self.args("--split", "holdout", "--label", "h"))
        self.assertEqual(code, 0, text)
        self.assertEqual(len(os.listdir(os.path.join(self.out, "h", "cases"))), 5)
        md = read(os.path.join(self.out, "h", "summary.md"))
        self.assertIn(frozen["freeze_id"], md)
        self.assertIn("holdout cases", md)
        # a second holdout run for the same freeze is refused
        code, text = run_main(self.args("--split", "holdout", "--label", "h2"))
        self.assertNotEqual(code, 0)
        self.assertIn("already run", text)
        # unless explicitly overridden, and then it is recorded as a repeat
        self.assertEqual(run_main(self.args("--split", "holdout", "--allow-repeat", "--label", "h3"))[0], 0)
        log = [json.loads(line) for line in read(ev.HOLDOUT_LOG).splitlines() if line.strip()]
        self.assertEqual([e["repeat"] for e in log], [False, True])

    def test_changing_the_prompt_after_the_freeze_blocks_the_holdout(self):
        run_main(self.args("--freeze"))
        with mock.patch.object(prompts, "COMPARE", prompts.COMPARE + "\n- a late tweak"):
            code, text = run_main(self.args("--split", "holdout"))
        self.assertNotEqual(code, 0)
        self.assertIn("no longer matches", text)
        self.assertIn("prompt_hash", text)

    def test_changing_the_labels_after_the_freeze_blocks_the_holdout(self):
        run_main(self.args("--freeze"))
        alt = os.path.join(self.tmp, "cases.csv")
        shutil.copy(os.path.join(HERE, "data", "test_cases.csv"), alt)
        with open(alt, "a", encoding="utf-8") as f:
            f.write("\n")
        code, text = run_main(self.args("--split", "holdout", "--cases", alt))
        self.assertNotEqual(code, 0)
        self.assertIn("test_cases_hash", text)

    def test_unverified_but_right_label_is_reported_as_label_only(self):
        real = pipeline.complete

        def liar(prompt, cfg, task=None, context=None):
            # fabricated report that claims same_sequence for every pitch
            return mock_complete("mock_faulty", "compare", context) if task == "compare" else real(prompt, cfg, task, context)

        with mock.patch.object(pipeline, "complete", liar):
            code, text = run_main(self.args("--label", "liar"))
        self.assertEqual(code, 0, text)
        s = jload(os.path.join(self.out, "liar", "summary.json"))["dev"]
        self.assertEqual(s["verified success (label right AND source checks passed)"].split("/")[0], "1")  # only the code-gated vague pitch
        self.assertEqual(s["sequence_overlap: intended show named AND verified"], "0/3")
        self.assertNotEqual(s["unverified / unsupported reports"].split("/")[0], "0")
        self.assertNotEqual(s["trope_only: false alarms (claimed same_sequence)"].split("/")[0], "0")

    def test_missing_key_stops_with_an_actionable_error_and_no_fake_summary(self):
        cfg = os.path.join(self.tmp, "cfg.json")
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump({"provider": "gemini", "model": "m", "api_key_env": "EVAL_TEST_NO_KEY", "retrieval": {"mode": "full"}}, f)
        os.environ.pop("EVAL_TEST_NO_KEY", None)
        code, text = run_main(self.args("--config", cfg, "--label", "nokey"))
        self.assertEqual(code, 3)
        self.assertIn("PROVIDER ERROR (missing_key)", text)
        self.assertIn("$env:EVAL_TEST_NO_KEY", text)
        self.assertFalse(os.path.exists(os.path.join(self.out, "nokey", "summary.md")))
        self.assertNotIn("MOCK", text)  # never silently falls back to the mock

    def test_guard_refuses_a_catalog_containing_a_test_pitch(self):
        bad = os.path.join(self.tmp, "cat.csv")
        with open(bad, "w", encoding="utf-8", newline="") as f:
            f.write("show_id,title,synopsis,source_url\n")
            f.write('SHOW_001,x,"%s",u\n' % case("T01")["pitch"].replace('"', '""'))
        code, text = run_main(self.args("--catalog", bad))
        self.assertNotEqual(code, 0)
        self.assertIn("inside the catalog", text)


if __name__ == "__main__":
    unittest.main()
