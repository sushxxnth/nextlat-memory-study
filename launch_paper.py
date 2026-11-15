"""Run/resume one paper-scale seed per method on an allocated CUDA GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
from run_pathstar import ROOT, verify_upstream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm',choices=['both','gpt','nextlat'],default='both')
    p.add_argument('--seed',type=int,default=1234)
    args=p.parse_args()
    verify_upstream()
    if not torch.cuda.is_available():
        raise SystemExit('Not launched: this process has no allocated CUDA GPU.')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('Not launched: the configured bf16 run requires a compatible CUDA GPU.')
    algorithms=['gpt','nextlat'] if args.algorithm=='both' else [args.algorithm]
    for algorithm in algorithms:
        out=ROOT/'runs'/f'{algorithm}_paper_seed{args.seed}'
        cfg=json.loads((ROOT/'configs/paper_g5_5.json').read_text());cfg['seed']=args.seed
        if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != cfg:
            raise SystemExit(f'Existing settings differ at {out}; refusing to reuse the run.')
        if (out/'results.json').exists():
            r=json.loads((out/'results.json').read_text())
            if r.get('status')=='completed' and r.get('steps')==cfg['steps']:
                print(f'Skipping completed {algorithm} seed {args.seed}.',flush=True)
                continue
        command=[sys.executable,str(ROOT/'run_pathstar.py'),'--config',str(ROOT/'configs/paper_g5_5.json'),
                 '--data-dir',str(ROOT/'datasets/paper_g5_5'),'--algorithm',algorithm,'--device','cuda',
                 '--seed',str(args.seed),'--out',str(out)]
        if (out/'checkpoint.pt').exists(): command.append('--resume')
        elif out.exists(): raise SystemExit(f'{out} exists without a checkpoint; inspect it before restarting.')
        print(f'Launching {algorithm}, seed {args.seed}, on {torch.cuda.get_device_name()}',flush=True)
        subprocess.run(command,check=True)


if __name__=='__main__':main()
