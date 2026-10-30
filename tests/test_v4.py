import io, json, unittest
import numpy as np
from core import simulate, SimulationConfig, run_analysis, evaluate_difference
from research import defaults
from theory import default_theory_inputs
from data_io import import_trial, TIMES
from copilot import make_context, ask, apply_edits, validate_reply

class V4Tests(unittest.TestCase):
    def setUp(self):
        self.df=simulate(SimulationConfig(n=500)); self.r=defaults(); self.i=default_theory_inputs()
        self.snap={'data':self.df,'analysis':run_analysis(self.df),'origin':'synthetic'}
        self.ctx=make_context(self.r,self.i,self.snap,False)
    def test_import_and_direction(self):
        clean=import_trial(self.df,0,'Higher is better',True)
        np.testing.assert_allclose(run_analysis(clean)['results'].estimate,self.snap['analysis']['results'].estimate)
        raw=self.df.copy(); raw['independent_competence']*=-1
        np.testing.assert_allclose(import_trial(raw,0,'Lower is better',True).independent_competence,self.df.independent_competence)
    def test_invalid_inputs(self):
        for mutate in [lambda d:d.assign(employee_id=1),lambda d:d.assign(stage2_guided=2),lambda d:d.assign(p1_observed=0),lambda d:d.assign(interim_skill=np.nan),lambda d:d.assign(high_progress=1-d.high_progress),lambda d:d.assign(p1_guided=.3)]:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): import_trial(mutate(self.df),0,'Higher is better',True)
        with self.assertRaises(ValueError): import_trial(self.df,0,'Higher is better')
    def test_time_order(self):
        d=self.df.copy()
        for j,k in enumerate(TIMES): d[k]=f'2026-01-0{j+1}'
        import_trial(d,0,'Higher is better')
        d['interim_time']='2026-01-05'
        with self.assertRaises(ValueError): import_trial(d,0,'Higher is better',True)
    def test_context_and_contrast(self):
        self.assertNotIn('employee_id',json.dumps(self.ctx)); self.assertNotIn('train_ids',json.dumps(self.ctx))
        self.assertIsNone(make_context(self.r,self.i,self.snap,True)['results'])
        a=self.snap['analysis']; rule=next(iter(a['policies'].values()))
        c=evaluate_difference(a['test'],rule,rule)
        self.assertEqual(c['difference'],0); self.assertEqual(c['se'],0)
    def test_edits(self):
        r,i=apply_edits(self.r,self.i,[{'field':'research.motivation','value':'Baseline absorptive capacity','reason':'Researcher request'}])
        self.assertNotEqual(r['motivation'],self.r['motivation'])
        for field,value in [('code','print(1)'),('inputs.response_threshold','nan'),('research.action1',self.r['action0'])]:
            with self.assertRaises(ValueError): apply_edits(self.r,self.i,[{'field':field,'value':value,'reason':'test'}])
    def test_protocols_mocked(self):
        reply={'answer':'Design review','edits':[],'source_ids':[],'limitations':[]}
        for provider,body in [('OpenAI API',{'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(reply)}]}]}),('Local Ollama',{'message':{'content':json.dumps(reply)}})]:
            seen=[]
            def opener(req,timeout):
                seen.append(json.loads(req.data)); return io.StringIO(json.dumps(body))
            result=ask(self.ctx,'Review',[],provider,'test-model','secret-key',opener)
            self.assertEqual(result['reply'],reply); self.assertNotIn('secret-key',json.dumps(result)); self.assertNotIn('employee_id',json.dumps(seen))
        with self.assertRaises(ValueError): validate_reply(reply|{'source_ids':['fabricated-doi']},self.ctx)
    def test_offline_and_missing_key(self):
        self.assertIn('randomize',ask(self.ctx,'Review',[],'Offline checklist')['reply']['answer'])
        with self.assertRaises(ValueError): ask(self.ctx,'Review',[],'OpenAI API','model')

class UploadUITests(unittest.TestCase):
    def test_upload_ui_and_export_origin(self):
        from pathlib import Path
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=40)
        at.checkbox(key='confirm_randomized').set_value(True)
        at.checkbox(key='confirm_timing').set_value(True)
        data=simulate(SimulationConfig(n=200,case='coping',augmentation=True,response_threshold=.5))
        data['independent_competence']=-data['independent_competence']
        csv=io.BytesIO(data.to_csv(index=False).encode())
        with patch('streamlit.file_uploader',return_value=csv):
            at.button(key='analyze_upload').click().run(timeout=40)
        self.assertEqual(len(at.exception),0,[x.message for x in at.exception])
        self.assertEqual(at.session_state['snapshot']['origin'],'uploaded')
        self.assertEqual(len(at.session_state['snapshot']['data']),200)
        at.button(key='review_copilot').click().run(timeout=40)
        self.assertEqual(at.session_state['assistant_proposal']['context']['results']['data_origin'],'uploaded')
        self.assertEqual(len(at.exception),0)
