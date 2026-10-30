import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from copilot import list_openai_models, safe_http_error

class ModelPickerTests(unittest.TestCase):
    def test_discovery_and_redacted_api_error(self):
        def opener(req, timeout):
            self.assertEqual(req.get_method(), 'GET')
            self.assertEqual(req.get_header('Authorization'), 'Bearer local-secret')
            return io.StringIO(json.dumps({'data': [{'id':'gpt-6-astra'}, {'id':'gpt-4.1-mini'}, {'id':'gpt-6-astra'}]}))
        self.assertEqual(list_openai_models('local-secret', opener), ['gpt-4.1-mini','gpt-6-astra'])
        error = urllib.error.HTTPError('https://api.openai.com', 404, 'not found', {},
            io.BytesIO(json.dumps({'error':{'message':'Model missing for local-secret sk-test123'}}).encode()))
        msg = safe_http_error(error, 'local-secret')
        self.assertIn('Model missing', msg)
        self.assertNotIn('local-secret', msg)
        self.assertNotIn('sk-test123', msg)

    def test_model_dropdown_and_account_reload(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=45)
        app.selectbox(key='copilot_provider').set_value('OpenAI API').run(timeout=45)
        self.assertEqual(app.selectbox(key='openai_model_choice').value, 'gpt-6-astra')
        app.text_input(key='copilot_key').set_value('secret-for-test').run(timeout=45)
        with patch('copilot.list_openai_models', return_value=['gpt-4.1-mini']):
            app.button(key='load_api_models').click().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='openai_model_choice').value, 'gpt-4.1-mini')
        app.text_input(key='copilot_key').set_value('different-secret').run(timeout=45)
        self.assertNotIn('openai_model_ids', app.session_state)
