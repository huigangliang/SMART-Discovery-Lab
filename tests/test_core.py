"""Targeted statistical invariants. Run: python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (OUTCOME, SimulationConfig, evaluate_difference, evaluate_policy,
                  fit_q_learning, run_analysis, simulate)


class StatisticalInvariantTests(unittest.TestCase):
    def test_reproducible_simulation_and_fixed_response_cutpoint(self):
        config = SimulationConfig(n=300, seed=17, response_threshold=0.75)
        a, b = simulate(config), simulate(config)
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(a.high_progress, (a.interim_skill >= 0.75).astype(int))

    def test_recorded_probabilities_follow_observed_actions_and_state(self):
        d = simulate(SimulationConfig(n=1000, p1_guided=0.7, p2_guided_low=0.2, p2_guided_high=0.8))
        np.testing.assert_allclose(d.p1_observed, np.where(d.stage1_guided == 1, 0.7, 0.3))
        p2 = np.where(d.high_progress == 1, d.stage1_guided, 0.2)
        np.testing.assert_allclose(d.p2_observed, np.where(d.stage2_guided == 1, p2, 1 - p2))

    def test_normalized_ipw_is_not_horvitz_thompson(self):
        d = simulate(SimulationConfig(n=40)).iloc[:4].copy()
        d['stage1_guided'], d['stage2_guided'] = 0, 0
        d.loc[d.index[0], ['stage1_guided', 'stage2_guided', OUTCOME]] = [1, 1, 10]
        d['high_progress'] = 0
        d['p1_observed'], d['p2_observed'] = 0.8, 0.25
        value = evaluate_policy(d, 1, 1)
        self.assertEqual(value['matched_n'], 1)
        self.assertEqual(value['estimate'], 10.0)  # HT would be 12.5.
        self.assertEqual(value['ess'], 1.0)
        self.assertTrue(np.isnan(value['se']))

    def test_nonconstant_path_weights_are_used(self):
        d = simulate(SimulationConfig(n=40)).iloc[:4].copy()
        d['stage1_guided'] = [1, 1, 0, 0]
        d['stage2_guided'] = [1, 1, 0, 0]
        d[OUTCOME] = [10.0, 30.0, 999.0, 999.0]
        d['high_progress'] = 0
        d['p1_observed'] = [0.5, 0.625, 0.5, 0.5]
        d['p2_observed'] = [0.5, 0.8, 0.5, 0.5]
        value = evaluate_policy(d, 1, 1)
        self.assertAlmostEqual(value['estimate'], 50 / 3)
        self.assertAlmostEqual(value['ess'], 1.8)

    def test_constant_outcome_and_identical_policy_difference(self):
        d = simulate(SimulationConfig(n=500, p1_guided=0.3, p2_guided_low=0.2, p2_guided_high=0.8))
        d[OUTCOME] = 7.25
        value = evaluate_policy(d, 1, 1)
        self.assertAlmostEqual(value['estimate'], 7.25)
        self.assertAlmostEqual(value['se'], 0.0)
        rule = (lambda x: np.ones(len(x), dtype=int), lambda x: np.ones(len(x), dtype=int))
        difference = evaluate_difference(d, rule, rule)
        self.assertEqual(difference['difference'], 0.0)
        self.assertEqual(difference['se'], 0.0)

    def test_no_matching_paths_returns_unavailable(self):
        d = simulate(SimulationConfig(n=40))
        d['stage1_guided'] = 0
        d['high_progress'] = 0
        d['p2_observed'] = .5
        value = evaluate_policy(d, 1, 1)
        self.assertEqual(value['matched_n'], 0)
        self.assertTrue(np.isnan(value['estimate']))
        self.assertEqual(value['ess'], 0.0)

    def test_known_linear_treatment_effects_recover_correct_actions(self):
        d = simulate(SimulationConfig(n=2000, seed=59, response_threshold=1000000))
        d[OUTCOME] = 2 + 0.7 * d.baseline_skill - 0.4 * d.motivation + 1.5 * d.stage1_guided - 2 * d.stage2_guided
        for theory_guided in (False, True):
            model = fit_q_learning(d, theory_guided)
            np.testing.assert_array_equal(model.recommend_stage1(d), np.ones(len(d), dtype=int))
            np.testing.assert_array_equal(model.recommend_stage2(d), np.zeros(len(d), dtype=int))
            np.testing.assert_allclose(model.q1(d, 1) - model.q1(d, 0), 1.5, atol=1e-10)
            np.testing.assert_allclose(model.q2(d, 1) - model.q2(d, 0), -2.0, atol=1e-10)

    def test_stage1_recommendation_uses_baseline_only(self):
        d = simulate(SimulationConfig(n=500))
        model = fit_q_learning(d)
        baseline_only = d[['baseline_skill', 'motivation']].copy()
        np.testing.assert_array_equal(model.recommend_stage1(d), model.recommend_stage1(baseline_only))
        changed = d.copy()
        changed['interim_skill'], changed['interim_motivation'], changed['high_progress'] = 10000, -10000, 1
        changed[OUTCOME] = 1e9
        np.testing.assert_array_equal(model.recommend_stage1(d), model.recommend_stage1(changed))

    def test_holdout_outcomes_never_affect_training(self):
        d = simulate(SimulationConfig(n=500))
        first = run_analysis(d)
        self.assertFalse(set(first['diagnostics']['train_ids']) & set(first['diagnostics']['test_ids']))
        changed = d.copy()
        changed.loc[changed.employee_id.isin(first['diagnostics']['test_ids']), OUTCOME] += 10000
        second = run_analysis(changed)
        for kind in ('unconstrained', 'theory_guided'):
            np.testing.assert_array_equal(first['models'][kind].beta1, second['models'][kind].beta1)
            np.testing.assert_array_equal(first['models'][kind].beta2, second['models'][kind].beta2)
        np.testing.assert_allclose(second['results'].estimate - first['results'].estimate, 10000)

    def test_null_scenario_removes_treatment_effects_from_states(self):
        # Same RNG draws and null mechanism: changing A1 randomization cannot
        # change intermediates or final outcomes when A2 randomization is held fixed.
        a = simulate(SimulationConfig(n=500, scenario='null', p1_guided=0.1))
        b = simulate(SimulationConfig(n=500, scenario='null', p1_guided=0.9))
        self.assertTrue((a.stage1_guided != b.stage1_guided).any())
        np.testing.assert_array_equal(a.interim_skill, b.interim_skill)
        np.testing.assert_array_equal(a.interim_motivation, b.interim_motivation)
        np.testing.assert_array_equal(a[OUTCOME], b[OUTCOME])

    def test_invalid_probabilities_and_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate(SimulationConfig(p1_guided=0.0))
        d = simulate(SimulationConfig())
        d.loc[0, 'p1_observed'] = 0.0
        with self.assertRaises(ValueError):
            evaluate_policy(d, 1, 1)


if __name__ == '__main__':
    unittest.main()
