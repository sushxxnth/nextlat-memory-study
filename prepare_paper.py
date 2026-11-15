"""Generate and audit the full official G(5,5) data without allocating a GPU."""
import json
from pathlib import Path
from run_pathstar import ROOT, make_data, verify_upstream

if __name__=='__main__':
    verify_upstream()
    c=json.loads((ROOT/'configs/paper_g5_5.json').read_text())
    tensors,lines=make_data(c,ROOT/'datasets/paper_g5_5')
    print(json.dumps({k:{'examples':len(v),'tokens_per_example':v.shape[1]} for k,v in tensors.items()},indent=2))
