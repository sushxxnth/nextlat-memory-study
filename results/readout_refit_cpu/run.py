from pathlib import Path
import hashlib,json,os
import torch
from threadpoolctl import threadpool_limits
from memory.readout_refit import refit_run
source=Path('outputs/nextlat-published/memory/runs/controls_v3_6000').resolve()
out=Path('outputs/readout_existing_cpu_v1').resolve()
plan=json.loads((source/'plan.json').read_text())
protocol={'scope':'Exploratory frozen full-vocabulary head intervention on existing CPU checkpoints; no backbone training; previously viewed test families. Same head fitting procedure as the new GPU follow-up.','source':str(source),'source_plan_sha256':hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),'method_source_sha256':hashlib.sha256(Path('/Users/sushanth.c/Documents/Codex/nextlat-memory-study/memory/readout_refit.py').read_bytes()).hexdigest(),'entries':[e['relative_run'] for e in plan['entries']]}
p=out/'protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:p.write_text(json.dumps(protocol,indent=2)+'\n')
torch.set_num_threads(2)
records=[]
with threadpool_limits(limits=2):
 for entry in plan['entries']:
  folder=out/entry['relative_run'];folder.mkdir(parents=True,exist_ok=True)
  for name in ['checkpoint.pt','families.json','probe_train_features.npz','test_features.npz']:
   target=folder/name
   if not target.exists():target.symlink_to(source/entry['relative_run']/name)
  r=refit_run(folder)
  s=json.loads((source/entry['relative_run']/'results.json').read_text())
  records.append({'condition':entry['condition'],'seed':entry['config']['seed'],'refit':r,'native_recorded':s['test']['64']['answer_accuracy'],'prequery_bits':s['test']['64']['probes']['before_query/stored_bits_mean']['accuracy']})
  v=r['test']['64']
  print(entry['relative_run'],v['native_cpu_redecode']['accuracy'],v['refitted']['accuracy'],flush=True)
(out/'all_results.json').write_text(json.dumps(records,indent=2)+'\n')
print('EXISTING_CPU_READOUT_STUDY_OK',flush=True)
