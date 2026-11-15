"""Reload final checkpoints and reproduce held-out features/predictions without fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from memory.experiment import L,build,fixed_data,verify_upstream
from memory.task import ANSWER_RED,ANSWER_BLUE


def recheck(root,delays=(8,12,64)):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());verify_upstream()
    torch.set_num_threads(2)
    records={}
    with threadpool_limits(limits=2):
        for entry in plan['entries']:
            folder=root/entry['relative_run'];c=entry['config']
            result=json.loads((folder/'results.json').read_text())
            assert result['environment']['device']=='cpu' and c['precision']=='32-true'
            state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert state['step']==c['steps'] and state['config']==c
            assert state['training_data_digest']==result['training_data_digest']
            fabric=L.Fabric(accelerator='cpu',devices=1,precision='32-true');fabric.launch()
            model=build(c,entry['algorithm'],fabric)
            model.model.module.load_state_dict(state['model']);model.eval()
            assert model.config.context_length==0
            checks={}
            with np.load(folder/'test_features.npz') as features:
                for delay in delays:
                    seq,_=fixed_data(c,'test',delay,c['eval_examples'])
                    logits=[];states=[]
                    for batch in seq.split(c['batch_size']):
                        with torch.inference_mode():
                            _,h=model.model(batch[:,:-2],return_hidden_states=True)
                            logits.append(model.model.module.lm_head(h[:,-1]).cpu())
                            states.append(h[:,-1].float().cpu().numpy())
                    logits=torch.cat(logits);h=np.concatenate(states)
                    np.testing.assert_array_equal(h,features[f'delay{delay}/at_query'])
                    predicted=logits.argmax(-1);targets=seq[:,-2]
                    accuracy=float((predicted==targets).float().mean())
                    restricted=float((logits[:,ANSWER_RED:ANSWER_BLUE+1].argmax(-1)==targets-ANSWER_RED).float().mean())
                    non_answer=float(((predicted!=ANSWER_RED)&(predicted!=ANSWER_BLUE)).float().mean())
                    part=result['test'][str(delay)]
                    assert abs(accuracy-part['answer_accuracy'])<1e-7
                    assert abs(restricted-part['constrained_answer_accuracy'])<1e-7
                    assert abs(non_answer-part['non_answer_token_rate'])<1e-7
                    checks[str(delay)]=dict(recall=accuracy,restricted_recall=restricted,
                                           non_answer_rate=non_answer,query_states_bitwise_equal=True)
            records[entry['relative_run']]=checks
    payload=dict(checked_runs=len(records),delays=list(delays),checked_run_delay_pairs=len(records)*len(delays),
                 scope='Full checkpoint reload and Transformer forward passes on original held-out test families, with answer/EOS excluded. No fitting or checkpoint selection.',
                 records=records)
    target=root/'summary/checkpoint_recheck.json';target.write_text(json.dumps(payload,indent=2)+'\n')
    print(target,flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    recheck(p.parse_args().run)
