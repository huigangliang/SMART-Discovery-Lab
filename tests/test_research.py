import io
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest
from research import defaults, candidate, local_sources, search_crossref, validate

class ResearchTests(unittest.TestCase):
    def test_unmatched_topic_does_not_invent_sources(self):
        s = {k:'zzzxqv' for k in defaults()}
        self.assertEqual(local_sources(s), [])

    def test_duplicate_interventions_rejected(self):
        s = defaults(); s['action1'] = s['action0']
        with self.assertRaises(ValueError): validate(s)

    def test_crossref_filter_dedup_and_provenance(self):
        payload = {'message':{'items':[
            {'DOI':'10.example/1','title':['Test title'],'container-title':['Organization Science'],'published':{'date-parts':[[2020]]}},
            {'DOI':'10.example/1','title':['Duplicate'],'container-title':['Organization Science']},
            {'DOI':'10.example/2','title':['Other'],'container-title':['Other Journal']}]}}
        def opener(req, timeout):
            self.assertIn('query.bibliographic=',req.full_url)
            self.assertNotIn('Authorization',req.headers)
            return io.StringIO(json.dumps(payload))
        rows=search_crossref('work learning','Organization Science',opener)
        self.assertEqual(len(rows),1)
        self.assertIn('not reviewed',rows[0]['status'])

    def test_custom_variables_draft_run_and_staleness(self):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=40)
        at.text_input(key='var_independent_competence').set_value('Employee exhaustion')
        at.text_input(key='var_action0').set_value('Usual scheduling')
        at.text_input(key='var_action1').set_value('Flexible scheduling')
        at.selectbox(key='var_direction').set_value('Lower is better')
        at.text_input(key='var_keywords').set_value('wellbeing autonomy')
        at.button(key='FormSubmitter:variables_form-Save variables and refresh references').click().run(timeout=40)
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertTrue(any('Settings have changed' in x.value for x in at.warning))
        self.assertNotEqual(at.session_state['snapshot']['research']['independent_competence'],'Employee exhaustion')
        at.button(key='generate_local').click().run(timeout=40)
        self.assertIn('Employee exhaustion', at.session_state['local_candidate']['research_question'])
        at.button(key='apply_local').click().run(timeout=40)
        self.assertFalse(at.exception, [e.message for e in at.exception])
        at.button(key='FormSubmitter:theory_form-Save hypotheses and rebuild design').click().run(timeout=40)
        at.button(key='run_pipeline').click().run(timeout=40)
        self.assertFalse(at.exception, [e.message for e in at.exception])
        snap=at.session_state['snapshot']
        self.assertEqual(snap['design']['outcome']['label'],'Employee exhaustion')
        self.assertEqual(snap['design']['outcome']['direction'],'Lower is better')
        self.assertEqual(snap['design']['stages'][0]['actions'][1]['label'],'Flexible scheduling')
        self.assertEqual(len(snap['analysis']['test']),200)

if __name__=='__main__': unittest.main()
