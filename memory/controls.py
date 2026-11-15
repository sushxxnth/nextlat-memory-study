"""Fixed 2x2 token-mask/delay-distribution control with fresh matched seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from memory.experiment import ROOT, validate, verify_upstream
from memory.sweep import subprocess_entry


TRAINING_SOURCES=['run_pathstar.py','memory/experiment.py','memory/task.py',
                  'memory/analysis.py','memory/sweep.py','memory/controls.py']


def make_plan(steps=2000,seeds=(1237,1238,1239),examples=1024):
    base=json.loads((ROOT/'memory/configs/pilot.json').read_text())
    base.update(num_cues=4,steps=steps,eval_every=500,eval_examples=examples,
                probe_examples_per_delay=examples,eval_delays=list(range(4,17))+[24,32,48,64],
                validation_delays=[4,8,16],probe_delays=[4,8,16],precision='32-true',
                scope='four-cue factorial delay-distribution and token-supervision controls; fixed final budget')
    entries=[]
    for distribution in ['sparse','dense']:
        delays=[4,8,16] if distribution=='sparse' else list(range(4,17))
        for mask in ['full','response']:
            for algorithm in ['gpt','nextlat']:
                condition=f'{distribution}_{mask}_{algorithm}'
                for seed in seeds:
                    c=dict(base,seed=seed,train_delays=delays,loss_scope=mask,
                           lambda_aux=0. if algorithm=='gpt' else 1.)
                    validate(c)
                    entries.append(dict(condition=condition,algorithm=algorithm,
                                        relative_run=f'{condition}/seed{seed}/{algorithm}',config=c))
    return dict(version=3,seeds=list(seeds),num_cues=4,in_distribution_gate=.95,
                factors=dict(delay_distribution=['sparse: 4,8,16','dense: every integer 4..16'],
                             token_supervision=['full sequence','ANSWER and EOS NTP/KL; context Smooth L1 retained']),
                probe_rule='All cells fit probes on the same separate families at delays 4,8,16; equal probe data budgets.',
                decision_rule='Report every cell. Require both methods >=95% recall at every trained delay; separately report common 4,8,16 gate.',
                checkpoint_rule='Final fixed-budget checkpoint; no test checkpoint selection.',
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in TRAINING_SOURCES},
                entries=entries)


def save_plan(root,plan):
    root.mkdir(parents=True,exist_ok=True)
    path=root/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('A different plan or training source already exists here; use a new output directory.')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    (root/'plan.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    for name in plan['source_sha256']:
        target=root/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=plan['source_sha256'][name]:
            raise ValueError('Stored source snapshot differs.')
        shutil.copy2(ROOT/name,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seeds',type=int,nargs='+',default=[1237,1238,1239])
    p.add_argument('--examples',type=int,default=1024)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();verify_upstream()
    if args.workers<1 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Positive workers and unique seeds required.')
    plan=make_plan(args.steps,args.seeds,args.examples);save_plan(args.out,plan)
    print(f'Fixed control plan: {len(plan["entries"])} runs, {args.steps} steps, fresh seeds {args.seeds}',flush=True)
    if args.plan_only:return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(subprocess_entry,e,args.out,'cpu') for e in plan['entries']]
        for i,f in enumerate(as_completed(futures),1):print(f'COMPLETED {i}/{len(futures)} {f.result()}',flush=True)
    from memory.control_report import report
    report(args.out)


if __name__=='__main__':main()
