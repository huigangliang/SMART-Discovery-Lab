"""Meaningful guardrails for the supported theory-to-SMART compiler."""

from copy import deepcopy
import json
import unittest

from theory import SOURCE_CATALOG, compile_design, default_theory_inputs, validate_design


class TheoryDesignTests(unittest.TestCase):
    def test_default_design_serializes_and_passes_structural_checks(self):
        design = compile_design(default_theory_inputs())
        json.dumps(design, allow_nan=False)
        self.assertFalse([c for c in design["checks"] if c["severity"] == "error"])
        self.assertFalse(design["provenance"]["ai_generated"])
        self.assertFalse(design["stages"][1]["rerandomize_all"])

    def test_late_variable_cannot_enter_stage1(self):
        inputs = default_theory_inputs()
        inputs["baseline_variables"].append("independent_competence")
        with self.assertRaises(ValueError):
            compile_design(inputs)
        design = compile_design(default_theory_inputs())
        design["stages"][0]["history"].append("interim_skill")
        self.assertTrue(any(c["severity"] == "error" for c in validate_design(design)))

    def test_impossible_probabilities_are_rejected(self):
        for p in [0, 1, -0.1, float("nan"), float("inf"), True, "0.5"]:
            with self.subTest(probability=p), self.assertRaises(ValueError):
                compile_design({}, {"p2_guided_low": p})

    def test_changed_threshold_and_probabilities_reach_executable_config(self):
        design = compile_design({"response_threshold": 0.7, "motivation_threshold": -0.3},
                                {"p1_guided": 0.3, "p2_guided_low": 0.6, "p2_guided_high": 0.4})
        self.assertEqual(design["executable_config"], {"response_threshold": 0.7, "motivation_threshold": -0.3,
                                                     "p1_guided": 0.3, "p2_guided_low": 0.6, "p2_guided_high": 0.4})
        self.assertEqual(design["stages"][1]["response_threshold"], 0.7)

    def test_researcher_text_is_not_claimed_as_published_result(self):
        hypothesis = "AI guidance could reduce learning under severe time pressure."
        design = compile_design({"proposition_scaffolding": hypothesis})
        card = next(x for x in design["theories"] if x["id"] == "scaffolding")
        self.assertEqual(card["researcher_hypothesis"], hypothesis)
        self.assertIn("untested", design["theories"][0]["evidence_level"])
        self.assertIn("Researcher-proposed", design["provenance"]["hypothesis_status"])

    def test_tampered_assignment_is_not_valid_smart(self):
        design = compile_design({})
        design["stages"][1]["randomized"] = False
        self.assertTrue(any(c["severity"] == "error" and c["check"] == "Stage-2 randomization" for c in validate_design(design)))

    def test_sources_resolve_and_keep_limits(self):
        ids = {s["id"] for s in SOURCE_CATALOG}
        self.assertEqual(len(ids), len(SOURCE_CATALOG))
        for source in SOURCE_CATALOG:
            self.assertTrue(source["doi"].startswith("10."))
            self.assertTrue(source["supports"])
            self.assertTrue(source["does_not_establish"])
            self.assertTrue(source["evidence_status"])
        design = compile_design({})
        for card in design["theories"]:
            self.assertTrue(set(card["source_ids"]) <= ids)

    def test_unsupported_custom_model_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            compile_design({"outcome_variable": "sales"})
        with self.assertRaises(ValueError):
            compile_design({"third_action": "withdraw all help"})
        with self.assertRaises(ValueError):
            compile_design({}, {"hidden_probability": 0.5})

    def test_defaults_do_not_share_mutable_lists(self):
        first, second = default_theory_inputs(), default_theory_inputs()
        first["enabled_theories"].clear()
        self.assertEqual(len(second["enabled_theories"]), 3)
        d1, d2 = compile_design({}), compile_design({})
        d1["source_catalog"][0]["title"] = "changed"
        self.assertNotEqual(d2["source_catalog"][0]["title"], "changed")


if __name__ == "__main__":
    unittest.main()
