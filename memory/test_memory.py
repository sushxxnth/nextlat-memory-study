import copy
import json
import unittest
import numpy as np
import torch
from memory import task
from memory.analysis import geometry, fit_probes, probe_metrics
from memory.experiment import ROOT, L, build, model_config, observe, validate


class MemoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.c=json.loads((ROOT/'memory/configs/pilot.json').read_text())
        cls.fabric=L.Fabric(accelerator='cpu',devices=1,precision='32-true');cls.fabric.launch()

    def test_family_balance_and_split_separation(self):
        families={}
        for split in task.SPLITS:
            seq,keys=task.batch(64,4,torch.Generator().manual_seed(99),split)
            families[split]=set(keys)
            self.assertEqual(seq.shape,(64,10))
            self.assertEqual(len(set(keys)),8)
            for g in seq.reshape(-1,task.GROUP,10):
                self.assertEqual(len(set(map(tuple,g[:,3:-3].tolist()))),1)
                self.assertEqual(set(map(tuple,g[:,1:3].tolist())),
                                 {(task.RED+cue,task.TAG0+tag) for cue in range(2) for tag in range(4)})
                self.assertTrue(torch.equal(g[:,-2],g[:,1]-task.RED+task.ANSWER_RED))
        for a in families:
            for b in families:
                if a!=b:self.assertFalse(families[a]&families[b])

    def test_shared_initialization_and_zero_auxiliary_gradients(self):
        c=dict(self.c,lambda_aux=0.)
        seq,_=task.batch(8,4,torch.Generator().manual_seed(2))
        a=build(c,'gpt',self.fabric);b=build(c,'nextlat',self.fabric)
        x=a.compute_loss(seq,backpropagate=True);y=b.compute_loss(seq,backpropagate=True)
        torch.testing.assert_close(x['loss'].squeeze(),y['loss'].squeeze(),atol=1e-6,rtol=1e-5)
        bp=dict(b.model.module.named_parameters())
        for key,p in a.model.module.named_parameters():
            torch.testing.assert_close(p,bp[key])
            torch.testing.assert_close(p.grad,bp[key].grad,atol=1e-5,rtol=1e-4)

    def test_future_answer_cannot_change_features_or_rollouts(self):
        seq,_=task.batch(8,8,torch.Generator().manual_seed(7),'test')
        changed=seq.clone();changed[:,-2:]=task.BOS
        for algorithm in ['gpt','nextlat']:
            model=build(self.c,algorithm,self.fabric);model.eval()
            a,ha=observe(model,seq,algorithm,'cpu');b,hb=observe(model,changed,algorithm,'cpu')
            for site in ha:np.testing.assert_array_equal(ha[site],hb[site])
            if algorithm=='nextlat':self.assertEqual(a['latent_rollout_steps'],8)

    def test_attention_only_used_once_for_latent_observation(self):
        model=build(self.c,'nextlat',self.fabric);model.eval()
        seq,_=task.batch(8,8,torch.Generator().manual_seed(8),'test')
        calls=[]
        hook=model.model.module.register_forward_pre_hook(lambda module,args:calls.append(args[0].shape[1]))
        observe(model,seq,'nextlat','cpu');hook.remove()
        self.assertEqual(calls,[12])  # prefix contains BOS/cue/tag/8 distractors/query

    def test_degenerate_predictor_is_rejected(self):
        with self.assertRaises(ValueError):validate(dict(self.c,proj_factor=.25))
        validate(self.c)

    def test_collapsed_geometry_is_finite(self):
        values=geometry(np.ones((16,128)))
        self.assertEqual(values['covariance_entropy_effective_rank'],0.)
        self.assertTrue(all(np.isfinite(v) for v in values.values()))

    def test_multicue_counterfactuals_and_query_targets(self):
        for cues in [4,8,16]:
            seq,keys=task.batch(256,8,torch.Generator().manual_seed(11),'test',cues)
            self.assertEqual(seq.shape,(256,8+cues+5))
            labels=task.labels(seq,cues)
            self.assertTrue(np.array_equal(labels['cue'],seq[:,-2].numpy()-task.ANSWER_RED))
            self.assertLess(int(seq.max()),task.vocab_size(cues))
            groups=seq.reshape(-1,8,seq.shape[1])
            for group in groups:
                query=int(group[0,-3])-task.VOCAB
                self.assertTrue(torch.equal(group[:,-3],group[0,-3].expand(8)))
                for j in range(cues):
                    expected=torch.tensor([task.RED]*4+[task.BLUE]*4) if j==query else group[0,1+j].expand(8)
                    self.assertTrue(torch.equal(group[:,1+j],expected))
            changed=seq.clone();changed[:,-2:]=task.BOS
            for key,value in task.labels(changed,cues).items():np.testing.assert_array_equal(value,labels[key])

    def test_multicue_split_separation(self):
        families={s:set(task.batch(128,4,torch.Generator().manual_seed(22),s,8)[1]) for s in task.SPLITS}
        for a in families:
            for b in families:
                if a!=b:self.assertFalse(families[a]&families[b])

    def test_multicue_no_answer_leak_and_no_attention_in_rollout(self):
        c=dict(self.c,num_cues=8)
        seq,_=task.batch(8,8,torch.Generator().manual_seed(8),'test',8)
        changed=seq.clone();changed[:,-2:]=task.BOS
        model=build(c,'nextlat',self.fabric);model.eval()
        calls=[]
        hook=model.model.module.register_forward_pre_hook(lambda module,args:calls.append(args[0].shape[1]))
        a,ha=observe(model,seq,'nextlat','cpu',8)
        b,hb=observe(model,changed,'nextlat','cpu',8)
        hook.remove()
        self.assertEqual(calls,[19,19])
        self.assertEqual(a['latent_rollout_steps'],8)
        for site in ha:np.testing.assert_array_equal(ha[site],hb[site])

    def test_multicue_zero_auxiliary_gradients(self):
        c=dict(self.c,num_cues=4,lambda_aux=0.)
        seq,_=task.batch(8,4,torch.Generator().manual_seed(33),'train',4)
        a=build(c,'gpt',self.fabric);b=build(c,'nextlat',self.fabric)
        x=a.compute_loss(seq,backpropagate=True);y=b.compute_loss(seq,backpropagate=True)
        torch.testing.assert_close(x['loss'].squeeze(),y['loss'].squeeze(),atol=1e-6,rtol=1e-5)
        bp=dict(b.model.module.named_parameters())
        for key,param in a.model.module.named_parameters():
            torch.testing.assert_close(param.grad,bp[key].grad,atol=1e-5,rtol=1e-4)

    def test_prequery_multicue_probe_does_not_require_unknown_query(self):
        parts={}
        for i,split in enumerate(['probe_train','test']):
            seq,_=task.batch(256,8,torch.Generator().manual_seed(100+i),split,4)
            labels=task.labels(seq,4)
            x=np.stack([labels[f'bit_{j}'] for j in range(4)],axis=1).astype(float)
            parts[split]=dict(labels=labels,features={'before_query':x,'at_query':x})
        probes=fit_probes({8:parts['probe_train']})
        self.assertNotIn(('before_query','cue'),probes)
        result=probe_metrics(probes,parts['test'])
        self.assertEqual(result['before_query/stored_bits_mean']['accuracy'],1.)
        self.assertEqual(result['before_query/selected_bit_given_query']['accuracy'],1.)

    def test_sweep_conditions_are_unique_and_match_planned_widths(self):
        from memory.sweep import make_plan
        plan=make_plan()
        self.assertEqual(len(plan['entries']),36)
        self.assertEqual(len({p['relative_run'] for p in plan['entries']}),36)
        for p in plan['entries']:
            c=p['config']
            width=128*round(2*c['width']*c['proj_factor']/128)
            if p['condition'].startswith('width'):
                self.assertEqual(width,int(p['condition'][5:]))
            else:self.assertEqual(width,128)


if __name__=='__main__':unittest.main(verbosity=2)
