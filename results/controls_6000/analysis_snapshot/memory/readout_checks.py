"""Audit saved output heads and features; no fitting or checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from memory.task import ANSWER_RED, ANSWER_BLUE


def audit(root):
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    records={}
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for entry in plan['entries']:
            folder=root/entry['relative_run']
            if not (folder/'results.json').exists():continue
            result=json.loads((folder/'results.json').read_text())
            state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            head=state['model']['lm_head.weight'].float()
            checks={}
            with np.load(folder/'test_features.npz') as features:
                for delay,part in result['test'].items():
                    h=torch.from_numpy(features[f'delay{delay}/at_query'])
                    # Retain the run's batch shape to minimize float32 GEMM differences.
                    with torch.inference_mode():
                        logits=torch.cat([chunk@head.T for chunk in h.split(entry['config']['batch_size'])])
                    cue=features[f'delay{delay}/cue']
                    target=cue+ANSWER_RED
                    predicted=logits.argmax(-1).numpy()
                    constrained=logits[:,ANSWER_RED:ANSWER_BLUE+1].argmax(-1).numpy()
                    correct=predicted==target
                    recalled=float(correct.mean())
                    if result['environment']['device']=='cpu':
                        assert abs(recalled-part['answer_accuracy'])<1e-7,(entry['relative_run'],delay)
                    item=dict(recomputed_recall=recalled,
                              recorded_recall=part['answer_accuracy'],
                              restricted_recall=float(np.mean(constrained==cue)),
                              outside_answer_token_rate=float(np.mean((predicted!=ANSWER_RED)&(predicted!=ANSWER_BLUE))),
                              atquery_linear_answer_probe=part['probes']['at_query/cue']['accuracy'],
                              predicted_token_counts={str(t):int(np.sum(predicted==t)) for t in np.unique(predicted)})
                    if f'delay{delay}/latent_query' in features:
                        latent=torch.from_numpy(features[f'delay{delay}/latent_query'])
                        with torch.inference_mode():
                            latent_logits=torch.cat([chunk@head.T for chunk in latent.split(entry['config']['batch_size'])])
                        latent_pred=latent_logits.argmax(-1).numpy()
                        latent_recall=float(np.mean(latent_pred==target))
                        if result['environment']['device']=='cpu':
                            assert abs(latent_recall-part['latent_rollout_accuracy'])<1e-7,(entry['relative_run'],delay,'latent')
                        item.update(latent_rollout_recomputed_recall=latent_recall,
                                    latent_rollout_recorded_recall=part['latent_rollout_accuracy'],
                                    latent_restricted_recall=float(np.mean(latent_logits[:,ANSWER_RED:ANSWER_BLUE+1].argmax(-1).numpy()==cue)),
                                    latent_outside_answer_token_rate=float(np.mean((latent_pred!=ANSWER_RED)&(latent_pred!=ANSWER_BLUE))))
                    if result['num_cues']>1:
                        query=features[f'delay{delay}/query_index']
                        item['per_query_index']={str(i):dict(examples=int(np.sum(query==i)),
                            recall=float(correct[query==i].mean()),
                            restricted_recall=float((constrained[query==i]==cue[query==i]).mean()))
                            for i in range(result['num_cues']) if np.any(query==i)}
                    checks[delay]=item
            records[entry['relative_run']]=checks
    output=root/'summary/readout_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(scope='Offline saved-state/output-head check, without fitting; CPU recall must match recorded metrics.',
                                     audited_runs=len(records),records=records),indent=2)+'\n')
    print(output)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run)
