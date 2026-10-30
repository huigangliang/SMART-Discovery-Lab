import copy
import unittest
from copilot import make_context, revision_question, prepare_revision, validate_reply
from research import defaults
from theory import default_theory_inputs

class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.r, self.i = defaults(), default_theory_inputs()
        self.context = make_context(self.r, self.i, None, False)
        self.p = {'provider':'Local Ollama', 'model':'mock', 'design_fingerprint':'design',
                  'run_created_at':'run1', 'context':self.context, 'reply':{'edits':[
                      {'field':'inputs.research_question','value':'When should support switch?', 'reason':'Explicit sequence question'},
                      {'field':'inputs.competing_prediction','value':'Continuing may outperform switching.', 'reason':'Alternative hypothesis'}]}}

    def test_subset_and_before_after_are_independent(self):
        before = copy.deepcopy(self.i)
        r,i,record = prepare_revision(self.r,self.i,self.p,['inputs.research_question'],'design','run1')
        self.assertEqual(self.i,before)
        self.assertEqual(i['research_question'],'When should support switch?')
        self.assertEqual(i['competing_prediction'],before['competing_prediction'])
        i['research_question']='Later edit'
        self.assertEqual(record['after']['inputs']['research_question'],'When should support switch?')
        self.assertEqual(record['before']['inputs'], before)

    def test_new_run_blocks_old_proposal_even_with_same_design(self):
        with self.assertRaises(ValueError):
            prepare_revision(self.r,self.i,self.p,['inputs.research_question'],'design','run2')

    def test_invalid_selected_subset_is_atomic(self):
        self.p['reply']['edits']=[{'field':'research.action0','value':self.r['action1'],'reason':'Collision'}]
        before = copy.deepcopy(self.r)
        with self.assertRaises(ValueError):
            prepare_revision(self.r,self.i,self.p,['research.action0'],'design','run1')
        self.assertEqual(self.r,before)

    def test_result_workflow_requires_current_results(self):
        with self.assertRaises(ValueError):
            revision_question('Plan next study from results','Revise hypothesis',self.context)
        self.context['results']={'data_origin':'synthetic'}
        prompt = revision_question('Plan next study from results','Revise hypothesis',self.context)
        self.assertIn('not empirical support',prompt)
        self.assertIn('independent validation',prompt)
        self.context['results_stale']=True
        with self.assertRaises(ValueError):
            revision_question('Plan next study from results','Revise hypothesis',self.context)

    def test_unsupported_augmentation_cannot_be_applied(self):
        reply={'answer':'Proposal','edits':[{'field':'inputs.augmentation','value':'true','reason':'Request'}],
               'source_ids':[], 'limitations':[]}
        with self.assertRaises(ValueError):
            validate_reply(reply,self.context)

if __name__ == '__main__':
    unittest.main()
