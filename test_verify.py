import copy
import unittest

from common import BY_ID, CLEAR_PITCH, good_report  # noqa: F401  (common also fixes sys.path)

from pitchcheck.verify import check_report, validate_extraction

SYN1 = BY_ID["SHOW_001"]["synopsis"]


def check(report, stats=None):
    return check_report(report, CLEAR_PITCH, BY_ID, stats)


def failed(c, key_part):
    return any(key_part in f for f in c["failures"])


class TestSequenceMatch(unittest.TestCase):
    def test_good_report_passes(self):
        c = check(good_report())
        self.assertTrue(c["ok"], c["failures"])
        self.assertEqual(c["counted_events"], 3)
        self.assertIn("does not certify story chronology", c["scope_note"])

    def test_unknown_show_fails(self):
        r = good_report()
        r["closest_id"] = "SHOW_999"
        self.assertTrue(failed(check(r), "exists in the catalog"))

    def test_invented_quote_fails_and_is_counted(self):
        r = good_report()
        r["shared_events"][1]["pitch_excerpt"] = "she secretly weds a billionaire stranger"
        c = check(r)
        self.assertFalse(c["ok"])
        self.assertLess(c["quotes_verified"], c["quotes_checked"])

    def test_quote_from_wrong_show_fails(self):
        r = good_report()
        r["shared_events"][1]["catalog_excerpt"] = "She enters a contract marriage with Ryan"  # SHOW_002 text
        self.assertTrue(failed(check(r), "occurs in the selected synopsis"))

    def test_text_order_failure(self):
        r = good_report()
        r["shared_events"][0], r["shared_events"][2] = r["shared_events"][2], r["shared_events"][0]
        self.assertTrue(failed(check(r), "same order"))

    def test_one_event_is_not_a_sequence(self):
        r = good_report()
        r["shared_events"] = r["shared_events"][:1]
        self.assertTrue(failed(check(r), "sequence match has at least two"))

    def test_two_events_of_the_same_kind_are_not_a_sequence(self):
        r = good_report()
        r["shared_events"] = r["shared_events"][:2]
        r["shared_events"][1]["kind"] = "opening_conflict"
        self.assertTrue(failed(check(r), "at least two kinds"))

    def test_duplicate_events_are_not_distinct(self):
        r = good_report()
        r["shared_events"][1] = copy.deepcopy(r["shared_events"][0])
        self.assertTrue(failed(check(r), "distinct"))

    def test_staged_versus_forced_marriage_is_not_the_same_event(self):
        r = good_report()
        # the pitch has her choose; the synopsis has the family force her
        r["shared_events"][1]["pitch_agency"] = "protagonist_chooses"
        r["shared_events"][1]["catalog_agency"] = "protagonist_forced"
        r["shared_events"][2]["pitch_agency"] = "protagonist_chooses"
        r["shared_events"][2]["catalog_agency"] = "other_acts"
        c = check(r)
        self.assertEqual(c["agency_mismatches"], [2, 3])
        self.assertFalse(c["ok"])  # only one counted event is left
        self.assertTrue(any("Who acts differs" in w for w in c["warnings"]))

    def test_mismatch_is_allowed_for_trope_only(self):
        r = good_report()
        r["verdict"] = "shared_trope_only"
        r["shared_events"] = r["shared_events"][1:2]
        r["shared_events"][0]["catalog_agency"] = "protagonist_forced"
        c = check(r)
        self.assertTrue(c["ok"], c["failures"])

    def test_missing_labels_fail(self):
        r = good_report()
        del r["shared_events"][0]["kind"]
        self.assertTrue(failed(check(r), "labels are valid"))
        r = good_report()
        r["order_basis"] = "maybe"
        self.assertTrue(failed(check(r), "order"))

    def test_unclear_order_is_acknowledged(self):
        r = good_report()
        r["order_basis"] = "unclear"
        c = check(r)
        self.assertTrue(c["ok"])
        self.assertTrue(any("do not clearly establish the order" in w for w in c["warnings"]))


class TestFirstDifference(unittest.TestCase):
    def test_established_needs_text_from_both_sides(self):
        r = good_report()
        r["first_difference"]["catalog_excerpt"] = None
        self.assertTrue(failed(check(r), "synopsis is quoted and found"))
        r = good_report()
        r["first_difference"]["pitch_excerpt"] = "something the pitch never says anywhere"
        self.assertTrue(failed(check(r), "pitch is quoted and found"))

    def test_established_needs_an_explanation(self):
        r = good_report()
        r["first_difference"]["explanation"] = "  "
        self.assertTrue(failed(check(r), "explanation"))

    def test_not_established_is_allowed_without_quotes_and_is_flagged(self):
        r = good_report()
        r["first_difference"] = {"established": False, "explanation": "The synopsis does not say how the stories diverge.",
                                 "pitch_excerpt": None, "catalog_excerpt": None}
        c = check(r)
        self.assertTrue(c["ok"], c["failures"])
        self.assertTrue(any("No first difference is established" in w for w in c["warnings"]))

    def test_difference_that_repeats_a_shared_event_fails(self):
        r = good_report()
        r["first_difference"]["pitch_excerpt"] = r["shared_events"][1]["pitch_excerpt"]
        r["first_difference"]["catalog_excerpt"] = r["shared_events"][1]["catalog_excerpt"]
        self.assertTrue(failed(check(r), "repeat a shared event"))

    def test_absolute_claim_about_a_blurb_is_warned(self):
        r = good_report()
        r["first_difference"]["explanation"] = "The existing show never has a testimony scene."
        c = check(r)
        self.assertTrue(any("absolute wording" in w for w in c["warnings"]))

    def test_difference_is_required_for_match_verdicts(self):
        r = good_report()
        r["first_difference"] = None
        self.assertTrue(failed(check(r), "first-difference note"))


class TestContradictionsAndTypes(unittest.TestCase):
    def test_no_match_with_a_show_or_events_is_rejected(self):
        c = check(dict(good_report(), verdict="no_meaningful_match"), {"events": 3, "words": 40})
        self.assertTrue(failed(c, "carries no matched show"))

    def test_insufficient_with_a_show_is_rejected(self):
        c = check(dict(good_report(), verdict="insufficient_information"))
        self.assertTrue(failed(c, "carries no matched show"))

    def test_insufficient_needs_one_specific_question(self):
        base = {"verdict": "insufficient_information", "closest_id": None, "shared_events": [], "first_difference": None}
        self.assertTrue(failed(check(dict(base, editor_question="")), "asks one specific question"))
        self.assertTrue(failed(check(dict(base, editor_question="What? Why?")), "asks one specific question"))
        self.assertTrue(check(dict(base, editor_question="What does she decide to do?"), {"events": 1, "words": 7})["ok"])

    def test_insufficient_on_a_detailed_pitch_is_not_accepted(self):
        base = {"verdict": "insufficient_information", "closest_id": None, "shared_events": [], "first_difference": None,
                "editor_question": "What does she decide?"}
        c = check(base, {"events": 4, "words": 40})
        self.assertTrue(failed(c, "thin enough"))

    def test_no_match_on_a_one_event_pitch_is_not_accepted(self):
        base = {"verdict": "no_meaningful_match", "closest_id": None, "shared_events": [], "first_difference": None}
        self.assertTrue(failed(check(base, {"events": 1, "words": 8}), "at least two stated events"))
        self.assertTrue(check(base, {"events": 2, "words": 30})["ok"])

    def test_wrong_json_types_never_crash(self):
        bad_reports = [
            None, [], "text", 5,
            {"verdict": 5},
            {"verdict": ["same_sequence"]},
            {"verdict": "same_sequence", "closest_id": 7, "shared_events": "oops", "first_difference": ["x"], "editor_question": 3},
            {"verdict": "same_sequence", "closest_id": "SHOW_001", "shared_events": [None, 5, "s", {"pitch_excerpt": 1, "catalog_excerpt": []}],
             "first_difference": {"established": "yes", "explanation": 9, "pitch_excerpt": {}, "catalog_excerpt": 4}, "order_basis": 3},
            {"verdict": "shared_trope_only", "closest_id": "SHOW_001", "shared_events": [{"pitch_excerpt": None}], "first_difference": {}},
            {"verdict": "no_meaningful_match", "closest_id": {}, "shared_events": {}, "first_difference": 1, "missing_info": "x"},
        ]
        for rep in bad_reports:
            c = check_report(rep, CLEAR_PITCH, BY_ID, {"events": 3, "words": 40})
            self.assertFalse(c["ok"], rep)
            self.assertIsInstance(c["failures"], list)


class TestExtractionValidation(unittest.TestCase):
    def test_wrong_shapes_are_not_well_formed(self):
        for data in (None, [], "x", {"events": "oops"}, {"events": None}, {}):
            self.assertFalse(validate_extraction(data, CLEAR_PITCH)["well_formed"], data)

    def test_invented_and_malformed_events_are_dropped_with_reasons(self):
        data = {"events": [
            {"role": "reveal", "text": "x", "excerpt": "totally absent from the source text at all"},
            "not an object",
            {"role": "reveal", "text": 5, "excerpt": "leaves her for another designer"},
            {"role": "opening_conflict", "text": "partner leaves", "excerpt": "leaves her for another designer"},
        ]}
        out = validate_extraction(data, CLEAR_PITCH)
        self.assertTrue(out["well_formed"])
        self.assertEqual(len(out["events"]), 1)
        self.assertEqual(len(out["dropped"]), 3)

    def test_overlapping_events_are_not_distinct(self):
        data = {"events": [
            {"role": "opening_conflict", "text": "a", "excerpt": "leaves her for another designer"},
            {"role": "other", "text": "b", "excerpt": "her fiance leaves her for another designer"},
        ]}
        out = validate_extraction(data, CLEAR_PITCH)
        self.assertEqual(len(out["events"]), 1)
        self.assertIn("overlaps", out["dropped"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
