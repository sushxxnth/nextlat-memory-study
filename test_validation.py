import json
import random
import unittest
import torch
from torch.nn import functional as F
from run_pathstar import ROOT, L, GPT, NextLat, config_for, build, encode, prepare_module, verify_upstream


def examples(c, count=4):
    lines=[]
    for i in range(count):
        random.seed(30000+i)
        edges,path,source,goal=prepare_module.star_or_sink_graph_maker(c['num_paths'],c['path_length'],c['max_nodes'],True)
        lines.append('|'.join(f'{a},{b}' for a,b in edges)+f'/{source},{goal}='+','.join(map(str,path)))
    return lines


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.c=json.loads((ROOT/'configs/pilot.json').read_text())
        cls.fabric=L.Fabric(accelerator='cpu',devices=1,precision='32-true');cls.fabric.launch()
        cls.data=encode(examples(cls.c),cls.c)

    def test_official_checkout_pinned_and_unmodified(self):
        verify_upstream()

    def test_official_graphs_and_tokenization(self):
        for line,seq in zip(examples(self.c),self.data):
            edges,tail=line.split('/');query,path=tail.split('=')
            source,goal=map(int,query.split(','));path=list(map(int,path.split(',')))
            edge_set={tuple(map(int,e.split(','))) for e in edges.split('|')}
            self.assertEqual(len(edge_set),20)
            self.assertEqual(path[0],source);self.assertEqual(path[-1],goal)
            self.assertEqual(len(path),5)
            self.assertTrue(all((a,b) in edge_set for a,b in zip(path,path[1:])))
            self.assertEqual(seq[-1].item(),104)
            self.assertEqual(seq[-6:-1].tolist(),path)
        self.assertEqual(self.data.shape[1],69)
        self.assertEqual(config_for(self.c,self.data).context_length,62)

    def test_causal_attention(self):
        model=GPT(config_for(self.c,self.data)).model.eval()
        altered=self.data.clone();altered[:,63:]=99
        with torch.no_grad():
            a=model(self.data);b=model(altered)
        torch.testing.assert_close(a[:,:63],b[:,:63],atol=1e-6,rtol=1e-5)

    def test_prompt_is_masked_from_ntp_loss(self):
        model=build(self.c,self.data,'gpt',self.fabric)
        got=model.compute_loss(self.data,backpropagate=False)['loss']
        logits=model.model(self.data[:,:-1])
        # Positions 62..67 predict source, four path nodes, then EOS.
        expected=F.cross_entropy(logits[:,62:].reshape(-1,106),self.data[:,63:].reshape(-1))
        torch.testing.assert_close(got,expected)

    def test_zero_auxiliary_matches_gpt_loss_and_gradients(self):
        c=dict(self.c,lambda_mse=0.,lambda_kl=0.)
        gpt=build(c,self.data,'gpt',self.fabric)
        nxt=build(c,self.data,'nextlat',self.fabric)
        a=gpt.compute_loss(self.data,backpropagate=True)
        b=nxt.compute_loss(self.data,backpropagate=True)
        torch.testing.assert_close(a['loss'].squeeze(),b['loss'].squeeze(),atol=1e-6,rtol=1e-5)
        gp=dict(gpt.model.module.named_parameters());np=dict(nxt.model.module.named_parameters())
        for key,p in gp.items():
            torch.testing.assert_close(p.grad,np[key].grad,atol=1e-5,rtol=1e-4)

    def test_auxiliary_targets_and_head_are_detached(self):
        model=NextLat(config_for(self.c,self.data,True))
        pred=torch.randn(2,6,128,requires_grad=True)
        target=torch.randn(2,6,128,requires_grad=True)
        teacher=torch.randn(2,5,106,requires_grad=True)
        tokens=torch.randint(0,100,(2,5))
        mse,ce,kl=model._nextlat_loss_function(pred,target,tokens,teacher,torch.ones(2,5,dtype=torch.bool),torch.ones(2,6,dtype=torch.bool))
        (mse+kl).backward()
        self.assertIsNone(target.grad);self.assertIsNone(teacher.grad)
        self.assertIsNone(model.model.lm_head.weight.grad)
        self.assertGreater(float(pred.grad.abs().sum()),0.)

    def test_nextlat_updates_dynamics_and_trunk(self):
        model=build(self.c,self.data,'nextlat',self.fabric)
        got=model.compute_loss(self.data,backpropagate=True)
        torch.testing.assert_close(got['loss'],got['next_token_loss']+got['mse_loss']+got['kl_loss'])
        for key in ['dynamics_model.mlp.0.weight','transformer.blocks.0.attn.c_attn.weight','token_embedding.weight']:
            grad=dict(model.model.module.named_parameters())[key].grad
            self.assertIsNotNone(grad);self.assertGreater(float(grad.abs().sum()),0.)

    def test_microbatches_preserve_effective_batch_gradients(self):
        from run_pathstar import backward_batch
        for algorithm in ['gpt','nextlat']:
            with self.subTest(algorithm=algorithm):
                full=build(self.c,self.data,algorithm,self.fabric)
                micro=build(self.c,self.data,algorithm,self.fabric)
                a=backward_batch(full,self.data,4)
                b=backward_batch(micro,self.data,2)
                torch.testing.assert_close(a['loss'],b['loss'],atol=1e-6,rtol=1e-5)
                for (_,pa),(_,pb) in zip(full.model.named_parameters(),micro.model.named_parameters()):
                    torch.testing.assert_close(pa.grad,pb.grad,atol=1e-5,rtol=1e-4)


if __name__=='__main__':unittest.main(verbosity=2)
