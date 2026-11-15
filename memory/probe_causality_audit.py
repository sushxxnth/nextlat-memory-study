"""Check that changing a future query cannot change the probed past state."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from memory.experiment import L, build, fixed_data, verify_upstream
from memory.task import VOCAB


def audit(root, out):
    root, out = Path(root), Path(out)
    plan=json.loads((root/'plan.json').read_text())
    torch.set_num_threads(2);verify_upstream()
    records=[]
    with threadpool_limits(limits=2), torch.inference_mode():
        for entry in plan['entries']:
            c=entry['config'];folder=root/entry['relative_run']
            state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            fabric=L.Fabric(accelerator='cpu',devices=1,precision='32-true');fabric.launch()
            model=build(c,entry['algorithm'],fabric)
            model.model.module.load_state_dict(state['model']);model.eval()
            seq,_=fixed_data(c,'test',64,32)
            prefix=seq[:,:-2]
            altered=prefix.clone()
            altered[:,-1]=VOCAB+(altered[:,-1]-VOCAB+1)%4
            _,before=model.model(prefix,return_hidden_states=True)
            _,after=model.model(altered,return_hidden_states=True)
            np.testing.assert_array_equal(before[:,-2].numpy(),after[:,-2].numpy())
            records.append(dict(run=entry['relative_run'],delay=64,examples=32,
                future_query_changed=True,before_query_state_bitwise_unchanged=True,
                query_state_changed=bool((before[:,-1]!=after[:,-1]).any())))
    result=dict(checked_models=len(records),query_perturbation_checks=len(records)*32,
                all_before_query_states_bitwise_unchanged=True,records=records)
    (out/'causal_query_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Passed {len(records)} models / {len(records)*32} future-query perturbations.")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit(args.run,args.out)
