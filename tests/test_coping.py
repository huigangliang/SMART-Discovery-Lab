import io
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from core import SimulationConfig, simulate, run_analysis, fixed_policies, evaluate_policy, fit_q_learning, pairwise_comparisons, validate_data, OUTCOME
from research import defaults, display_frame, restore_engine_columns, compile_research, local_sources
from theory import default_theory_inputs, compile_design, SOURCE_CATALOG
from data_io import import_trial
from visuals import tree_figure, smart_tree, figure_bytes

class CopingTests(unittest.TestCase):
    def setUp(self):
        self.c=SimulationConfig(n=2000,seed=11,case='coping',augmentation=True,response_threshold=.5)
        self.d=simulate(self.c)

    def test_response_and_all_eight_paths(self):
        np.testing.assert_allclose(self.d.interim_skill,self.d.baseline_skill-self.d.interim_stress)
        np.testing.assert_array_equal(self.d.high_progress,(self.d.interim_skill>=.5).astype(int))
        self.assertEqual(len(self.d.groupby(['stage1_guided','high_progress','stage2_guided'])),8)
        r=self.d.high_progress==1
        np.testing.assert_array_equal(self.d.loc[r,'stage2_guided'],self.d.loc[r,'stage1_guided'])
        np.testing.assert_allclose(self.d.loc[r,'p2_observed'],1)
        np.testing.assert_allclose(self.d.loc[~r,'p2_observed'],1/3)
        self.assertTrue(self.d[OUTCOME].between(-10,0).all())

    def test_six_strategies_with_shared_responder_hand_values(self):
        d=self.d.iloc[:8].copy()
        d['stage1_guided']=[0]*4+[1]*4
        d['high_progress']=[1,0,0,0]*2
        d['stage2_guided']=[0,0,1,2,1,1,0,2]
        d['p1_observed']=.5
        d['p2_observed']=[1,1/3,1/3,1/3]*2
        d[OUTCOME]=np.arange(1,9,dtype=float)
        policies=fixed_policies(augmentation=True)
        self.assertEqual(len(policies),6)
        for rule,expected in zip(policies.values(),[(2+6*2)/8,(2+6*3)/8,(2+6*4)/8,(10+6*6)/8,(10+6*7)/8,(10+6*8)/8]):
            self.assertAlmostEqual(evaluate_policy(d,*rule)['estimate'],expected)

    def test_q2_treats_combination_categorically(self):
        d=simulate(replace(self.c,response_threshold=100))
        # Non-monotone effects: B worse than A, combination best. Ordinal slope cannot recover this.
        d[OUTCOME]=2+.3*d.baseline_skill+1.2*d.stage1_guided-2*(d.stage2_guided==1)+3*(d.stage2_guided==2)
        for guided in (False,True):
            m=fit_q_learning(d,guided)
            np.testing.assert_array_equal(m.recommend_stage2(d),2)
            np.testing.assert_array_equal(m.recommend_stage1(d),1)
            np.testing.assert_allclose(m.q2(d,1)-m.q2(d,0),-2,atol=1e-10)
            np.testing.assert_allclose(m.q2(d,2)-m.q2(d,0),3,atol=1e-10)

    def test_null_removes_all_support_and_dose_effects(self):
        a=simulate(replace(self.c,scenario='null',p1_guided=.1))
        b=simulate(replace(self.c,scenario='null',p1_guided=.9,augmentation_dose='added_time',augmentation_effect=1.5,switch_effect=1))
        np.testing.assert_array_equal(a.interim_skill,b.interim_skill)
        np.testing.assert_array_equal(a[OUTCOME],b[OUTCOME])

    def test_dose_and_combination_parameters_reach_only_combination(self):
        a=self.d; b=simulate(replace(self.c,augmentation_effect=.7))
        noncombo=a.stage2_guided!=2
        np.testing.assert_array_equal(a.loc[noncombo,OUTCOME],b.loc[noncombo,OUTCOME])
        self.assertTrue((b.loc[~noncombo,OUTCOME]>a.loc[~noncombo,OUTCOME]).any())
        more=simulate(replace(self.c,augmentation_dose='added_time'))
        self.assertTrue((more.loc[~noncombo,OUTCOME]!=a.loc[~noncombo,OUTCOME]).any())

    def test_coping_csv_roundtrip_and_bad_reduction(self):
        raw=self.d.copy();raw[OUTCOME]=-raw[OUTCOME]
        shown=display_frame(raw,defaults())
        loaded=restore_engine_columns(pd.read_csv(io.StringIO(shown.to_csv(index=False))),defaults())
        d=import_trial(loaded,.5,'Lower is better',True,augmentation=True,coping=True)
        np.testing.assert_allclose(d[OUTCOME],self.d[OUTCOME])
        self.assertEqual(d.attrs['stage2_actions'],[0,1,2])
        loaded.loc[0,'interim_skill']+=1
        with self.assertRaises(ValueError):import_trial(loaded,.5,'Lower is better',True,augmentation=True,coping=True)

    def test_missing_training_arm_not_silently_removed(self):
        d=self.d[self.d.stage2_guided!=2].copy()
        with self.assertRaises(ValueError):fit_q_learning(d)

    def test_augmented_holdout_and_holm_family(self):
        a=run_analysis(self.d);self.assertEqual(len(a['results']),8)
        self.assertEqual(len(pairwise_comparisons(a)),28)
        changed=self.d.copy()
        changed.loc[changed.employee_id.isin(a['diagnostics']['test_ids']),OUTCOME]+=1
        b=run_analysis(changed)
        for k in a['models']:
            np.testing.assert_array_equal(a['models'][k].beta2,b['models'][k].beta2)
        np.testing.assert_allclose(b['results'].estimate-a['results'].estimate,1)

    def test_theory_and_visuals_match_coping(self):
        d=compile_research(compile_design(default_theory_inputs()),defaults(),True,'equal_time')
        self.assertEqual([x['code'] for x in d['stages'][1]['actions']],[0,1,2])
        self.assertEqual(d['embedded_strategy_count'],6)
        self.assertIn('30/30',d['protocol']['combined_dose'])
        self.assertEqual({x['id'] for x in local_sources(defaults())},{x['id'] for x in SOURCE_CATALOG})
        self.assertNotIn('scaffold',' '.join(x['title'].lower() for x in SOURCE_CATALOG))
        self.assertIn('Path 8',smart_tree(defaults(),True))
        self.assertIn(b'<svg',figure_bytes(tree_figure(defaults(),True)))

class CopingUITests(unittest.TestCase):
    def test_default_coping_augmentation_and_dosage_rerun(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=45)
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertFalse(at.error,[e.value for e in at.error])
        snap=at.session_state['snapshot']
        self.assertEqual(snap['research']['action0'],'Problem-focused coping support')
        self.assertEqual(snap['research']['direction'],'Lower is better')
        self.assertEqual(len(snap['analysis']['results']),8)
        self.assertEqual(snap['design']['observed_path_count'],8)
        at.selectbox(key='dosage').set_value('added_time').run(timeout=45)
        self.assertTrue(any('Settings have changed' in x.value for x in at.warning))
        at.button(key='run_pipeline').click().run(timeout=45)
        self.assertIn('120 minutes',at.session_state['snapshot']['design']['protocol']['combined_dose'])
        at.button(key='review_copilot').click().run(timeout=45)
        self.assertIn('120 minutes',at.session_state['assistant_proposal']['reply']['answer'])
        at.checkbox(key='augmentation').uncheck().run(timeout=45)
        at.button(key='run_pipeline').click().run(timeout=45)
        self.assertEqual(len(at.session_state['snapshot']['analysis']['results']),6)
        self.assertFalse(at.exception,[e.message for e in at.exception])

if __name__=='__main__':unittest.main()
