"""Exercise complete Streamlit flows, not just import/syntax checks."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class IntegratedAppTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(str(APP)).run(timeout=40)
        self.assertEqual(len(at.exception), 0, [x.message for x in at.exception])
        return at

    def test_default_integrated_run(self):
        at = self.app()
        snapshot = at.session_state["snapshot"]
        self.assertEqual(len(at.tabs), 6)
        self.assertEqual(len(snapshot["data"]), 500)
        self.assertEqual(len(snapshot["analysis"]["train"]), 300)
        self.assertEqual(len(snapshot["analysis"]["test"]), 200)
        self.assertEqual(len(snapshot["analysis"]["results"]), 8)
        self.assertEqual(set(snapshot["analysis"]["models"]), {"unconstrained", "theory_guided"})

    def test_changed_design_and_simulation_propagate_together(self):
        at = self.app()
        at.checkbox(key="augmentation").uncheck().run(timeout=40)
        at.slider(key="n").set_value(800)
        at.selectbox(key="scenario").set_value("reversed")
        at.slider(key="p1").set_value(0.7)
        at.slider(key="p2_low").set_value(0.25)
        at.slider(key="response_threshold").set_value(0.8)
        at.run(timeout=40)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["snapshot"]["data"]), 500)
        self.assertTrue(any("Settings have changed" in x.value for x in at.warning))
        at.button(key="run_pipeline").click().run(timeout=40)
        self.assertEqual(len(at.exception), 0, [x.message for x in at.exception])
        snap = at.session_state["snapshot"]
        self.assertEqual(len(snap["data"]), 800)
        self.assertEqual(snap["config"]["scenario"], "reversed")
        self.assertEqual(snap["design"]["executable_config"]["response_threshold"], 0.8)
        self.assertEqual(snap["design"]["executable_config"]["p1_guided"], 0.7)
        self.assertEqual(snap["design"]["executable_config"]["p2_guided_low"], 0.25)
        self.assertTrue(((snap["data"]["interim_skill"] >= 0.8).astype(int) == snap["data"]["high_progress"]).all())

    def test_researcher_hypothesis_is_preserved_in_run(self):
        at = self.app()
        text = "Competing prediction: direct solutions improve later transfer under high task complexity."
        at.text_area(key="edit_competing_prediction").set_value(text)
        at.button(key="FormSubmitter:theory_form-Save hypotheses and rebuild design").click().run(timeout=40)
        at.button(key="run_pipeline").click().run(timeout=40)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["snapshot"]["design"]["hypothesis_tests"]["competing_prediction"], text)

    def test_assistant_offline_and_missing_credentials(self):
        at = self.app()
        at.button(key="review_copilot").click().run(timeout=40)
        self.assertEqual(len(at.exception), 0, [x.message for x in at.exception])
        self.assertIn("randomize", at.session_state["assistant_proposal"]["reply"]["answer"])
        at.selectbox(key="copilot_provider").set_value("OpenAI API").run(timeout=40)
        at.selectbox(key="openai_model_choice").set_value("Other / custom model ID").run(timeout=40)
        at.text_input(key="copilot_model").set_value("test-model")
        at.button(key="review_copilot").click().run(timeout=40)
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("API key" in x.value for x in at.error))

    def test_assistant_edits_propagate_and_invalidate(self):
        at = self.app()
        at.button(key="review_copilot").click().run(timeout=40)
        result=at.session_state["assistant_proposal"]
        result["reply"]["edits"]=[{"field":"research.motivation","value":"Baseline absorptive capacity","reason":"Requested construct"}]
        at.session_state["assistant_proposal"]=result
        at.run(timeout=40)
        at.button(key="apply_copilot").click().run(timeout=40)
        self.assertEqual(len(at.exception),0,[x.message for x in at.exception])
        self.assertEqual(at.text_input(key="var_motivation").value,"Baseline absorptive capacity")
        self.assertTrue(any("Settings have changed" in x.value for x in at.warning))
        at.button(key="run_pipeline").click().run(timeout=40)
        self.assertEqual(at.session_state["snapshot"]["research"]["motivation"],"Baseline absorptive capacity")

class RevisionWorkflowUITests(unittest.TestCase):
    def test_guided_proposal_subset_application_and_audit(self):
        from unittest.mock import patch
        at = AppTest.from_file(str(APP)).run(timeout=40)
        at.selectbox(key="copilot_provider").set_value("Local Ollama").run(timeout=40)
        at.text_input(key="copilot_model").set_value("mock-model")
        at.text_area(key="revision_goal").set_value("Clarify coping hypotheses")
        at.run(timeout=40)
        reply = {"reply":{"answer":"Exploratory coping proposal", "edits":[
            {"field":"inputs.research_question", "value":"When should coping support switch?", "reason":"Sequence"},
            {"field":"inputs.competing_prediction", "value":"Continuing may work better", "reason":"Alternative"}],
            "source_ids":[], "limitations":["Not empirical evidence"]},
            "provider":"Local Ollama", "model":"mock-model", "created_at":"mock-revision"}
        with patch("copilot.ask", return_value=reply) as mocked:
            at.button(key="propose_revision").click().run(timeout=40)
            self.assertIn("Researcher-approved revision workflow", mocked.call_args.args[1])
        self.assertEqual(len(at.exception),0,[x.message for x in at.exception])
        old_question=at.session_state["inputs"]["research_question"]
        self.assertNotEqual(old_question,"When should coping support switch?")
        at.multiselect(key="revision_fields_mock-revision").set_value(["inputs.research_question"])
        at.button(key="apply_copilot").click().run(timeout=40)
        self.assertEqual(len(at.exception),0,[x.message for x in at.exception])
        self.assertEqual(at.session_state["inputs"]["research_question"],"When should coping support switch?")
        record=at.session_state["revision_history"][-1]
        self.assertEqual(len(record["applied_edits"]),1)
        self.assertEqual(record["before"]["inputs"]["research_question"],old_question)
        self.assertTrue(any("Settings have changed" in x.value for x in at.warning))


if __name__ == "__main__":
    unittest.main()
