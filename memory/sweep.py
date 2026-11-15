"""Run a fixed-plan one-factor sweep and a multiple-cue memory study.

python -m memory.sweep --out memory/runs/controlled_v2 --device cpu
All results are final-budget tests, never used to choose a checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from threadpoolctl import threadpool_limits
from memory.experiment import ROOT, validate, run, verify_upstream


def conditions(profile='all'):
    core = [('ntp', 'gpt', 1, 0., .5),
            ('lambda01', 'nextlat', 1, .1, .5),
            ('lambda1', 'nextlat', 1, 1., .5),
            ('lambda10', 'nextlat', 1, 10., .5),
            ('width256', 'nextlat', 1, 1., 1.),
            ('width512', 'nextlat', 1, 1., 2.)]
    capacity = [(f'cues{k}_{algorithm}', algorithm, k, 0. if algorithm == 'gpt' else 1., .5)
                for k in [4, 8, 16] for algorithm in ['gpt', 'nextlat']]
    return core if profile == 'core' else capacity if profile == 'capacity' else core+capacity


def make_plan(steps=2000, seeds=(1234, 1235, 1236), examples=1024, profile='all', precision='32-true'):
    base=json.loads((ROOT/'memory/configs/pilot.json').read_text())
    base.update(steps=steps, eval_every=500, eval_examples=examples,
                probe_examples_per_delay=examples, precision=precision,
                eval_delays=[4, 8, 12, 16, 24, 32, 48, 64],
                scope='controlled auxiliary-weight, predictor-capacity and memory-load study; final-budget test')
    entries=[]
    for name, algorithm, cues, weight, factor in conditions(profile):
        for seed in seeds:
            c=dict(base, seed=seed, num_cues=cues, lambda_aux=weight, proj_factor=factor)
            validate(c)
            entries.append(dict(condition=name, algorithm=algorithm,
                                relative_run=f'{name}/seed{seed}/{algorithm}', config=c))
    return dict(version=2, seeds=list(seeds), profile=profile,
                in_distribution_gate=.95,
                decision_rule='Interpret delay-shift differences only when both matched models reach >=95% recall at every training delay. Report all runs regardless.',
                checkpoint_rule='Fixed final training budget; test once after training. Do not select by test accuracy.',
                entries=entries)


def subprocess_entry(entry, root, device):
    out=root/entry['relative_run']
    out.parent.mkdir(parents=True,exist_ok=True)
    config=out.parent/'condition_config.json'
    config.write_text(json.dumps(entry['config'],indent=2)+'\n')
    command=[sys.executable,'-m','memory.experiment','--config',str(config.resolve()),
             '--out',str(out.parent.resolve()),'--algorithm',entry['algorithm'],'--device',device]
    if (out/'checkpoint.pt').exists():command.append('--resume')
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(ROOT))
    with (out.parent/'execution.log').open('a') as stream:
        subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return entry['relative_run']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seeds', type=int, nargs='+', default=[1234,1235,1236])
    p.add_argument('--examples', type=int, default=1024)
    p.add_argument('--profile', choices=['core','capacity','all'], default='all')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--workers',type=int,default=1,help='Independent CPU processes; CUDA requires one worker.')
    args=p.parse_args()
    verify_upstream()
    if len(set(args.seeds)) != len(args.seeds): raise ValueError('Seeds must be unique.')
    if args.workers<1 or (args.device=='cuda' and args.workers!=1):raise ValueError('Use positive CPU workers; one worker per allocated GPU.')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('No allocated CUDA GPU.')
    precision='bf16-mixed' if args.device=='cuda' else '32-true'
    plan=make_plan(args.steps, args.seeds, args.examples, args.profile, precision)
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('This output directory already has a different immutable plan.')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    (args.out/'plan.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    print(f'Plan: {len(plan["entries"])} runs; {args.steps} steps; device={args.device}',flush=True)
    if args.plan_only:return
    torch.set_num_threads(2)
    if args.workers>1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(subprocess_entry,entry,args.out,args.device) for entry in plan['entries']]
            for index,future in enumerate(as_completed(futures),1):
                print(f'COMPLETED {index}/{len(futures)} {future.result()}',flush=True)
    else:
        with threadpool_limits(limits=2):
            for index, entry in enumerate(plan['entries'],1):
                out=args.out/entry['relative_run']
                print(f'RUN {index}/{len(plan["entries"])} {entry["relative_run"]}',flush=True)
                out.parent.mkdir(parents=True,exist_ok=True)
                run(entry['config'],entry['algorithm'],out,args.device,(out/'checkpoint.pt').exists())
    from memory.sweep_report import report
    report(args.out)


if __name__=='__main__':main()
