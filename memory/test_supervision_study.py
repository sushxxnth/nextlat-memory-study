"""Check the frozen-head intervention and fixed-budget study design."""
import unittest
import numpy as np
import torch
from memory.readout_refit import fit_full_head, head_metrics
from memory.supervision_study import make_plan


class SupervisionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_factorial_plan_matches_data_and_evaluates_all_training_delays(self):
        plan=make_plan()
        self.assertEqual(len(plan['entries']),40)
        self.assertEqual(plan['stages'],[500,2000,6000])
        self.assertEqual(len({(e['condition'],e['config']['seed']) for e in plan['entries']}),40)
        for seed in plan['seeds']:
            configs=[e['config'] for e in plan['entries'] if e['config']['seed']==seed]
            reference={k:v for k,v in configs[0].items() if k not in ['lambda_aux','loss_scope']}
            for c in configs:
                self.assertEqual(reference,{k:v for k,v in c.items() if k not in ['lambda_aux','loss_scope']})
                self.assertTrue(set(c['train_delays']) <= set(c['eval_delays']))
                self.assertEqual(c['precision'],'32-true')
                self.assertEqual(c['num_cues'],4)

    def test_full_vocab_refit_can_fix_wrong_token_without_changing_features(self):
        # A nuisance token dominates the native 24-class head, although the
        # answer is perfectly available in frozen features. No forced choice.
        x=np.tile(np.array([[1.,-1.],[1.,1.]]),(64,1))
        y=np.tile(np.array([6,7]),64)
        original=np.zeros((24,2));original[3,0]=5.
        before_x=x.copy();before_w=original.copy()
        baseline=head_metrics(x,y,torch.as_tensor(original))
        fitted,audit=fit_full_head(x,y,original,max_iter=100)
        self.assertEqual(baseline['non_answer_rate'],1.)
        result=head_metrics(x,y,fitted)
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['non_answer_rate'],0.)
        self.assertEqual(tuple(fitted.shape),(24,2))
        np.testing.assert_array_equal(x,before_x)
        np.testing.assert_array_equal(original,before_w)
        self.assertLess(audit['gradient_max'],1e-4)

    def test_bad_stage_order_and_duplicate_seeds_rejected(self):
        for stages in [(2000,500),(500,500),(0,500)]:
            with self.assertRaises(ValueError):make_plan(stages=stages)
        with self.assertRaises(ValueError):make_plan(seeds=[1,1])


if __name__=='__main__':
    unittest.main()
