"""Check response masks and fair factorial comparisons, not expected research outcomes."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from memory import task
from memory.controls import make_plan
from memory.experiment import ROOT,L,build,loss_context,validate,fixed_data,run


class ControlChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base=json.loads((ROOT/'memory/configs/pilot.json').read_text())
        cls.base.update(num_cues=4,loss_scope='response')
        cls.fabric=L.Fabric(accelerator='cpu',devices=1,precision='32-true');cls.fabric.launch()

    def test_response_ntp_is_exactly_answer_and_eos(self):
        for delay in [4,16]:
            seq,_=task.batch(8,delay,torch.Generator().manual_seed(50),num_cues=4)
            model=build(self.base,'gpt',self.fabric)
            with loss_context(model,seq,self.base) as boundary:
                self.assertEqual(boundary,seq.shape[1]-3)
                actual=model.compute_loss(seq,backpropagate=False)['loss']
                logits=model.model(seq[:,:-1])
                expected=F.cross_entropy(logits[:,-2:].reshape(-1,24),seq[:,-2:].reshape(-1))
                torch.testing.assert_close(actual,expected)
            self.assertEqual(model.config.context_length,0)

    def test_zero_auxiliary_response_gradients_match_ntp(self):
        c=dict(self.base,lambda_aux=0.)
        seq,_=task.batch(8,8,torch.Generator().manual_seed(55),num_cues=4)
        a=build(c,'gpt',self.fabric);b=build(c,'nextlat',self.fabric)
        with loss_context(a,seq,c),loss_context(b,seq,c):
            x=a.compute_loss(seq,backpropagate=True);y=b.compute_loss(seq,backpropagate=True)
        torch.testing.assert_close(x['loss'].squeeze(),y['loss'].squeeze(),atol=1e-6,rtol=1e-5)
        bp=dict(b.model.module.named_parameters())
        for name,param in a.model.module.named_parameters():
            torch.testing.assert_close(param.grad,bp[name].grad,atol=1e-5,rtol=1e-4)

    def test_kl_mask_excludes_prompt_but_latent_loss_keeps_context(self):
        seq,_=task.batch(8,8,torch.Generator().manual_seed(56),num_cues=4)
        model=build(self.base,'nextlat',self.fabric)
        with patch.object(model,'_categorical_kl_loss',wraps=model._categorical_kl_loss) as kl, \
             patch.object(model,'_nextlat_loss_function',wraps=model._nextlat_loss_function) as latent:
            with loss_context(model,seq,self.base):
                values=model.compute_loss(seq,backpropagate=True)
            self.assertEqual(len(kl.call_args_list),3)
            for call in kl.call_args_list:
                mask=call.args[2]
                self.assertEqual(int(mask.sum()),16)
                self.assertTrue(bool(mask[:,-2:].all()))
                self.assertFalse(bool(mask[:,:-2].any()))
            for call in latent.call_args_list:
                self.assertTrue(bool(call.kwargs['mse_mask'][:,:-2].any()))
            self.assertGreater(float(values['mse_loss']),0.)
            grad=model.model.module.dynamics_model.mlp[0].weight.grad
            self.assertIsNotNone(grad);self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_context_restores_even_on_failure(self):
        seq,_=task.batch(8,4,torch.Generator().manual_seed(57),num_cues=4)
        model=build(self.base,'gpt',self.fabric)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with loss_context(model,seq,self.base):raise RuntimeError('fixture')
        self.assertEqual(model.config.context_length,0)

    def test_plan_is_crossed_and_probe_data_is_identical(self):
        plan=make_plan()
        self.assertEqual(len(plan['entries']),24)
        self.assertEqual(len({e['relative_run'] for e in plan['entries']}),24)
        fixed={}
        for e in plan['entries']:
            c=e['config'];self.assertEqual(c['probe_delays'],[4,8,16]);self.assertEqual(c['validation_delays'],[4,8,16])
            self.assertEqual(c['train_delays'],[4,8,16] if e['condition'].startswith('sparse') else list(range(4,17)))
            for split in ['test','probe_train']:
                seq,keys=fixed_data(c,split,8,64)
                key=c['seed'],split
                if key in fixed:
                    torch.testing.assert_close(seq,fixed[key][0]);self.assertEqual(keys,fixed[key][1])
                else:fixed[key]=seq,keys

    def test_full_scope_preserves_legacy_loss_and_invalid_scopes_reject(self):
        seq,_=task.batch(8,4,torch.Generator().manual_seed(58),num_cues=4)
        c=dict(self.base,loss_scope='full');model=build(c,'gpt',self.fabric)
        with loss_context(model,seq,c):actual=model.compute_loss(seq,backpropagate=False)['loss']
        expected=model.compute_loss(seq,backpropagate=False)['loss']
        torch.testing.assert_close(actual,expected)
        with self.assertRaises(ValueError):validate(dict(c,loss_scope='invalid'))
        with self.assertRaises(ValueError):validate(dict(c,probe_delays=[64]))

    def test_continuation_matches_uninterrupted_training_exactly(self):
        from memory.extend_controls import prepare_checkpoint
        c=dict(self.base,steps=2,eval_every=2,eval_examples=128,probe_examples_per_delay=128,
               train_delays=[4],eval_delays=[4],validation_delays=[4],probe_delays=[4],seed=1237)
        longer=dict(c,steps=4)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            run(c,'nextlat',root/'first','cpu',False)
            with self.assertRaises(ValueError):
                prepare_checkpoint(root/'first',root/'bad',dict(longer,loss_scope='full'))
            original=(root/'first/checkpoint.pt').read_bytes()
            prepare_checkpoint(root/'first',root/'continued',longer)
            self.assertEqual(original,(root/'first/checkpoint.pt').read_bytes())
            run(longer,'nextlat',root/'continued','cpu',True)
            run(longer,'nextlat',root/'fresh','cpu',False)
            a=torch.load(root/'continued/checkpoint.pt',weights_only=False)
            b=torch.load(root/'fresh/checkpoint.pt',weights_only=False)
            self.assertEqual(a['training_data_digest'],b['training_data_digest'])
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            self.assertEqual([r['step'] for r in a['logs']],[1,2,3,4])


if __name__=='__main__':unittest.main(verbosity=2)
