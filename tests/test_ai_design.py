import json
import unittest
from unittest.mock import patch
from io import BytesIO
from urllib.error import HTTPError

from ai_design import AIDesignError, generate_draft, validate_draft


def candidate():
    return {
        "title": "Example", "research_question": "Does guidance improve independent transfer?",
        "proposition_learning": "Experience may develop skill.",
        "proposition_scaffolding": "Guidance may be conditional on interim skill.",
        "proposition_motivation": "Training motivation may moderate guidance.",
        "competing_prediction": "Guidance could reduce independent learning through reliance.",
        "boundary_conditions": "Individual employees doing the simulated task.",
        "response_threshold": 0.0, "motivation_threshold": 0.0,
        "decision_rationale": "Prespecified interim measurement precedes randomization.",
        "source_ids": ["verified"], "proposed_extensions": ["AI-specific effects remain hypotheses."],
        "falsification_plan": "Compare null and contrary mechanisms.",
        "limitations": ["Synthetic data cannot validate the hypotheses."],
    }


class DraftBoundaryTests(unittest.TestCase):
    def test_unknown_citation_rejected(self):
        d = candidate(); d["source_ids"] = ["invented"]
        with self.assertRaises(AIDesignError):
            validate_draft(d, ["verified"])

    def test_nonfinite_threshold_rejected(self):
        d = candidate(); d["response_threshold"] = float("nan")
        with self.assertRaises(AIDesignError):
            validate_draft(d, ["verified"])

    def test_missing_credentials_never_make_request(self):
        with patch("ai_design.urllib.request.urlopen") as request:
            with self.assertRaises(AIDesignError):
                generate_draft("question", {}, [{"id": "verified"}], "", "")
            request.assert_not_called()

    def test_completed_structured_response(self):
        body = {"status": "completed", "id": "mock_response", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(candidate())}]}]}
        with patch("ai_design.urllib.request.urlopen", return_value=BytesIO(json.dumps(body).encode())) as request:
            result = generate_draft("question", {}, [{"id": "verified"}], "test-key", "test-model")
        self.assertEqual(result["draft"]["source_ids"], ["verified"])
        sent = json.loads(request.call_args.args[0].data)
        self.assertFalse(sent["store"])
        self.assertNotIn("test-key", json.dumps(sent))

    def test_refusal_does_not_become_design(self):
        body = {"status": "completed", "output": [{"content": [{"type": "refusal", "refusal": "no"}]}]}
        with patch("ai_design.urllib.request.urlopen", return_value=BytesIO(json.dumps(body).encode())):
            with self.assertRaises(AIDesignError):
                generate_draft("question", {}, [{"id": "verified"}], "test-key", "test-model")

    def test_http_error_redacts_body_and_key(self):
        err = HTTPError("https://api.openai.com/v1/responses", 401, "SECRET_BODY", {}, None)
        with patch("ai_design.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(AIDesignError) as raised:
                generate_draft("question", {}, [{"id": "verified"}], "SECRET_KEY", "model")
        self.assertNotIn("SECRET", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
