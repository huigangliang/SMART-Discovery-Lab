import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from core import SimulationConfig, simulate, run_analysis, fixed_policies, evaluate_policy, pairwise_comparisons, validate_data
from research import defaults, display_frame, restore_engine_columns
from data_io import import_trial
from copilot import ask, make_context
from visuals import smart_tree, comparison_figure, figure_bytes

class V5ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.df=simulate(SimulationConfig(n=2000,seed=91))

    def test_six_paths_and_four_complete_strategies(self):
        self.assertEqual(len(self.df.groupby(['stage1_guided','high_progress','stage2_guided'])),6)
        rules=fixed_policies(); self.assertEqual(len(rules),4)
        r=self.df[self.df.high_progress==1]
        np.testing.assert_array_equal(r.stage1_guided,r.stage2_guided)
        np.testing.assert_array_equal(r.p2_observed,1)
        for first,second in rules.values():
            np.testing.assert_array_equal(second(r),first(r))

    def test_responder_shared_weights_hand_calculation(self):
        d=self.df.iloc[:6].copy()
        d['stage1_guided']=[0,0,0,1,1,1]
        d['high_progress']=[1,0,0,1,0,0]
        d['stage2_guided']=[0,0,1,1,1,0]
        d['p1_observed']=.5; d['p2_observed']=[1,.5,.5,1,.5,.5]
        d['independent_competence']=[10,20,30,40,50,60]
        rules=list(fixed_policies().values())
        # Responders weight 2; matching nonresponders weight 4.
        expected=[(20+80)/6,(20+120)/6,(80+200)/6,(80+240)/6]
        for rule,y in zip(rules,expected):
            self.assertAlmostEqual(evaluate_policy(d,*rule)['estimate'],y)

    def test_bad_responder_assignments_and_probability_rejected(self):
        idx=self.df.index[self.df.high_progress==1][0]
        for col,value in [('stage2_guided',1-self.df.loc[idx,'stage1_guided']),('p2_observed',.5)]:
            d=self.df.copy(); d.loc[idx,col]=value
            with self.assertRaises(ValueError): validate_data(d)

    def test_q_learning_only_optimizes_eligible_decisions(self):
        a=run_analysis(self.df)
        r=self.df[self.df.high_progress==1]
        for model in a['models'].values():
            np.testing.assert_array_equal(model.recommend_stage2(r),r.stage1_guided)
            self.assertNotIn('high_progress',model.describe()['stage2_features'])
        self.assertEqual(len(a['results']),6)

    def test_unadjusted_pairwise_reference(self):
        import math
        c=pairwise_comparisons(run_analysis(self.df))
        self.assertEqual(len(c),15)
        self.assertNotIn('p_holm',c.columns)
        for row in c.itertuples():
            self.assertAlmostEqual(row.p_raw,math.erfc(abs(row.difference/row.se)/math.sqrt(2)))
            self.assertEqual(row.significance != 'ns',row.p_raw < .05)

    def test_custom_label_csv_roundtrip(self):
        spec=defaults(); spec['interim_skill']='Week 4 performance'; spec['motivation']='Baseline commitment'
        shown=display_frame(self.df,spec)
        self.assertIn('Week 4 performance',shown)
        self.assertNotIn('interim_skill',shown)
        self.assertFalse(any('[' in c for c in shown))
        reread=pd.read_csv(io.StringIO(shown.to_csv(index=False)))
        restored=restore_engine_columns(reread,spec)
        clean=import_trial(restored,0,'Higher is better',True)
        np.testing.assert_allclose(clean.independent_competence,self.df.independent_competence)

    def test_chart_and_tree_export(self):
        a=run_analysis(self.df); row=pairwise_comparisons(a).iloc[0].to_dict()
        spec=defaults(); spec['direction']='Higher is better'
        fig=comparison_figure(a['results'],row['policy_a'],row['policy_b'],row,spec)
        self.assertEqual(len(fig.axes[0].patches),2)
        np.testing.assert_allclose([x.get_height() for x in fig.axes[0].patches],a['results'].estimate.iloc[:2])
        self.assertIn(b'<svg',figure_bytes(fig))
        tree=smart_tree(defaults())
        for i in range(1,7): self.assertIn('Path '+str(i),tree)

class V5UI(unittest.TestCase):
    def test_send_and_review_are_separate_and_conversation_retained(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key='ask_copilot').disabled)
        app.button(key='review_copilot').click().run(timeout=45)
        self.assertEqual(len(app.session_state['conversation']),2)
        app.selectbox(key='copilot_provider').set_value('OpenAI API').run(timeout=45)
        self.assertFalse(app.button(key='ask_copilot').disabled)
        app.selectbox(key='openai_model_choice').set_value('Other / custom model ID').run(timeout=45)
        app.text_input(key='copilot_model').set_value('test-model')
        app.text_input(key='copilot_key').set_value('test-secret')
        app.text_area(key='copilot_question').set_value('Explain that previous review')
        captured=[]
        def fake_ask(context,question,history,provider,model,key):
            captured.append((question,list(history)))
            return {'reply':{'answer':'Continuation answer','edits':[],'source_ids':[],'limitations':[]},'provider':provider,'model':model}
        with patch('copilot.ask',fake_ask): app.button(key='ask_copilot').click().run(timeout=45)
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertEqual(captured[0][0],'Explain that previous review')
        self.assertEqual(len(captured[0][1]),2)
        self.assertNotIn('test-secret',json.dumps(app.session_state['assistant_audit']))
        self.assertEqual(len(app.session_state['conversation']),4)

    def test_policy_selection_updates_contrast_and_chart(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=45)
        names=list(app.session_state['snapshot']['analysis']['policies'])
        app.selectbox(key='contrast_a').set_value(names[0])
        app.selectbox(key='contrast_b').set_value(names[1]).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['current_contrast']['policy_a'],names[0])
        self.assertIn('p_raw',app.session_state['current_contrast'])
        self.assertTrue(any(b.label=='Download comparison figure (SVG)' for b in app.get('download_button')))
