"""Fixed-budget progression study separating readout failure and cue accessibility."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from threadpoolctl import threadpool_limits
from memory.experiment import ROOT, run, validate, verify_upstream
from memory.extend_controls import prepare_checkpoint
from memory.readout_refit import refit_run

SOURCES = ['run_pathstar.py','memory/task.py','memory/analysis.py','memory/experiment.py',
           'memory/extend_controls.py','memory/readout_refit.py','memory/supervision_study.py']


def make_plan(seeds=(1240,1241,1242,1243,1244), stages=(500,2000,6000), examples=1024):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct training seeds.')
    if not stages or list(stages) != sorted(set(stages)) or stages[0] < 1:
        raise ValueError('Stages must be strictly increasing positive budgets.')
    base = json.loads((ROOT/'memory/configs/pilot.json').read_text())
    base.update(num_cues=4, precision='32-true', train_delays=list(range(4,17)),
        eval_delays=list(range(4,17))+[24,32,48,64], validation_delays=[4,8,16], probe_delays=[4,8,16],
        eval_examples=examples, probe_examples_per_delay=examples, steps=stages[-1],
        eval_every=500, scope='Exploratory matched GPU progression and full-vocabulary readout-refit study')
    entries=[]
    for seed in seeds:
        for loss_scope in ['full','response']:
            for algorithm, weight, name in [('gpt',0.,'ntp'),('nextlat',1.,'nextlat1'),
                                           ('nextlat',.1,'nextlat01'),('nextlat',10.,'nextlat10')]:
                config=dict(base, loss_scope=loss_scope, lambda_aux=weight, seed=seed)
                validate(config)
                entries.append(dict(condition=f'{loss_scope}_{name}', algorithm=algorithm,
                    relative_run=f'{loss_scope}_{name}/seed{seed}/{algorithm}',config=config))
    return dict(version=1, seeds=list(seeds), stages=list(stages), entries=entries,
        precision='float32; TF32 disabled on CUDA; not bitwise-equivalent to CPU',
        primary='Final fixed budget, delay 64. Compare native recall, non-answer rate, full-vocabulary head-refit recall, and frozen before-query cue decoding.',
        progression='Report every specified budget and evaluated delay; never choose the best test checkpoint.',
        learning_gate='>=95% native recall at every trained delay, evaluated separately at the final budget. Do not discard failed seeds.',
        inference='Refit operates on at-query states after full attention; it cannot establish pre-query memory. Low probe/refit scores do not prove erasure.',
        data_caveat='New training seeds and fixed data-selection rule. Noise-family partitions are unchanged and can overlap prior evaluation families, especially at short delays; this is exploratory, not a fresh held-out confirmation.',
        head_refit='Native 24-class bias-free head, initialized from saved weights, frozen backbone, separate probe families at 4,8,16, fixed regularization and optimizer. Test labels never used for fitting.',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES})


def save_plan(out, plan):
    out.mkdir(parents=True,exist_ok=True)
    path=out/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing plan/source differs; use a fresh output directory.')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    for name in plan['source_sha256']:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)


def execute(out, plan, device):
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('No allocated CUDA GPU; no training launched.')
    verify_upstream(); torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    env=dict(device=device,torch=str(torch.__version__),started_unix=time.time(),
             gpu=torch.cuda.get_device_name() if device=='cuda' else None,tf32=False)
    (out/'execution_environment.json').write_text(json.dumps(env,indent=2)+'\n')
    with threadpool_limits(limits=2):
        for stage_index, step in enumerate(plan['stages']):
            for entry in plan['entries']:
                previous=(out/f'step_{plan["stages"][stage_index-1]}'/entry['relative_run']) if stage_index else None
                config=dict(entry['config'],steps=step)
                folder=out/f'step_{step}'/entry['relative_run']
                if not (folder/'results.json').exists():
                    if previous is not None and not (folder/'checkpoint.pt').exists():
                        prepare_checkpoint(previous,folder,config)
                run(config,entry['algorithm'],folder,device,(folder/'checkpoint.pt').exists())
                if json.loads((folder/'results.json').read_text())['steps'] != step:
                    raise RuntimeError('Unexpected checkpoint budget.')
                refit_run(folder)
                print(f'STUDY_PROGRESS {entry["relative_run"]} step={step}',flush=True)
                # Finish each budget across all cells before moving to the next.
                completed=[e['relative_run'] for e in plan['entries']
                           if (out/f'step_{plan["stages"][-1]}'/e['relative_run']/'readout_refit.json').exists()]
                stage_done=sum((out/f'step_{step}'/e['relative_run']/'readout_refit.json').exists() for e in plan['entries'])
                (out/'status.json').write_text(json.dumps(dict(completed=len(completed),planned=len(plan['entries']),
                    current_budget=step,completed_at_current_budget=stage_done,
                    status='completed' if len(completed)==len(plan['entries']) else 'running',runs=completed),indent=2)+'\n')
    print('SUPERVISION_STUDY_OK',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=[1240,1241,1242,1243,1244])
    p.add_argument('--stages',type=int,nargs='+',default=[500,2000,6000])
    p.add_argument('--examples',type=int,default=1024)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();plan=make_plan(args.seeds,args.stages,args.examples)
    save_plan(args.out,plan)
    print(f'{len(plan["entries"])} models; budgets {plan["stages"]}; device={args.device}',flush=True)
    if not args.plan_only: execute(args.out,plan,args.device)
