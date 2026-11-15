"""Continue every factorial cell to one common longer budget; preserve originals."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import torch
from memory.controls import make_plan,save_plan
from memory.experiment import ROOT,verify_upstream
from memory.sweep import subprocess_entry


def prepare_checkpoint(source,out,config):
    """Adapt budget metadata in a NEW checkpoint; tensors and RNG stay intact."""
    old=json.loads((source/'config.json').read_text())
    old_without_steps={k:v for k,v in old.items() if k!='steps'}
    new_without_steps={k:v for k,v in config.items() if k!='steps'}
    if old_without_steps!=new_without_steps or config['steps']<=old['steps']:
        raise ValueError('Continuation may increase only the fixed step budget.')
    result=json.loads((source/'results.json').read_text())
    if result['status']!='completed' or result['steps']!=old['steps']:
        raise ValueError('Continuation requires a completed source run.')
    if out.exists():
        if not (out/'continuation.json').exists() or json.loads((out/'config.json').read_text())!=config:
            raise ValueError('Existing continuation is incomplete or inconsistent; inspect it.')
        return
    state=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    if state['config']!=old or state['step']!=old['steps'] or state['training_data_digest']!=result['training_data_digest']:
        raise ValueError('Source checkpoint does not match its result metadata.')
    out.mkdir(parents=True,exist_ok=False)
    metadata=dict(source=str(source.resolve()),source_steps=old['steps'],new_budget=config['steps'],
                  source_checkpoint_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),
                  rule='Only config.steps metadata is changed in the copied checkpoint. Model/optimizer/data RNG/logs are retained. Original files are untouched.')
    state['config']=config
    torch.save(state,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if (source/'validation.jsonl').exists():shutil.copy2(source/'validation.jsonl',out/'validation.jsonl')
    (out/'continuation.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();verify_upstream()
    if args.workers<1:raise ValueError('Positive workers required.')
    previous=json.loads((args.source/'plan.json').read_text())
    # The decision uses validation at common trained delays, never OOD scores.
    validation=[]
    for entry in previous['entries']:
        folder=args.source/entry['relative_run']
        if not (folder/'results.json').exists():raise ValueError('Complete every source condition first.')
        last=json.loads((folder/'validation.jsonl').read_text().splitlines()[-1])
        validation.append(dict(run=entry['relative_run'],minimum_common_validation_recall=min(p['answer_accuracy'] for p in last['delays'].values())))
    plan=make_plan(args.steps,previous['seeds'],previous['entries'][0]['config']['eval_examples'])
    plan['continuation']=dict(source_plan_sha256=hashlib.sha256((args.source/'plan.json').read_bytes()).hexdigest(),
        source_budget=previous['entries'][0]['config']['steps'],
        rationale=f'Several common-delay validation recalls are below 95%. Extend EVERY cell to a common {args.steps}-step budget, not a best-test checkpoint.',
        validation_at_source_budget=validation)
    plan['source_sha256']['memory/extend_controls.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_plan(args.out,plan)
    previous_entries={e['relative_run']:e for e in previous['entries']}
    for entry in plan['entries']:
        old=previous_entries[entry['relative_run']]
        if {k:v for k,v in old['config'].items() if k!='steps'}!={k:v for k,v in entry['config'].items() if k!='steps'}:
            raise ValueError('Factor/configuration mismatch.')
        prepare_checkpoint(args.source/entry['relative_run'],args.out/entry['relative_run'],entry['config'])
    print(f'Continuing all {len(plan["entries"])} cells to {args.steps} steps.',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(subprocess_entry,e,args.out,'cpu') for e in plan['entries']]
        for i,f in enumerate(as_completed(futures),1):print(f'COMPLETED {i}/{len(futures)} {f.result()}',flush=True)
    from memory.control_report import report
    report(args.out)


if __name__=='__main__':main()
