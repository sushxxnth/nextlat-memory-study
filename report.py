"""Summarize completed published-task pilots without treating them as a replication."""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

records=[]
fig,axes=plt.subplots(1,2,figsize=(10,4))
for algorithm,label in [('gpt','GPT / NTP'),('nextlat','NextLat')]:
    d=ROOT/'runs'/f'{algorithm}_pilot_seed1234'
    result=json.loads((d/'results.json').read_text())
    assert result['status']=='completed'
    with (d/'training.csv').open() as f: logs=list(csv.DictReader(f))
    val=[json.loads(line) for line in (d/'validation.jsonl').read_text().splitlines()]
    axes[0].plot([int(r['step']) for r in logs],[float(r['next_token_loss']) for r in logs],label=label,alpha=.8)
    axes[1].plot([r['step'] for r in val],[r['official_sampled_path_accuracy'] for r in val],'o-',label=label)
    result['initial_ntp_loss']=float(logs[0]['next_token_loss'])
    result['final_batch_ntp_loss']=float(logs[-1]['next_token_loss'])
    records.append(result)
axes[0].set(xlabel='Optimization step',ylabel='Path + EOS NTP cross-entropy',title='Training losses (same minibatch order)')
axes[1].set(xlabel='Optimization step',ylabel='Full-path exact-match rate',title='Official sampled validation evaluation',ylim=(-.02,1.02))
for ax in axes: ax.legend(frameon=False)
fig.suptitle('Path-Star G(5,5): reduced-model pilot, one seed',fontsize=13)
fig.tight_layout();fig.savefig(ROOT/'pilot_comparison.png',dpi=180);plt.close(fig)
text='''# Published Path-Star validation status

**Aditi & Sushanth — Science of Deep Learning**

We now use the authors' official Path-Star generator, tokenizer, GPT architecture,
NextLat objective, and autoregressive sampled full-path evaluator. All are imported
from the pinned, unmodified NextLat checkout at commit
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`.

The local runner handles data caching, matching transformer initialization and
minibatches, training, checkpointing, and supplementary greedy evaluation. The
original custom delayed-cue code is not used by these experiments.

## What has been validated

Eight checks passed: source provenance; graph/path/token correctness; causal masking;
prompt-masked prediction loss; zero-auxiliary NextLat/GPT loss and gradient equality;
detached auxiliary targets and head; and active gradients through NextLat's dynamics
and transformer; and microbatch/full-batch loss and gradient agreement for both
models. These are implementation checks, not a confirmation of published
benchmark accuracy.

## Completed pilot results

Both arms use G(5,5), 100 node labels, 4,096 fixed training graphs, 256 separate
validation graphs, 512 held-out test graphs, a 2-layer/128-wide/4-head transformer,
batch 32, and 300 steps. One seed (1234); CPU float32. NextLat uses horizon 3,
Smooth L1 weight 1, KL weight 1, and projection factor 0.5. Identical transformer
initial weights and minibatch orders are used in both arms.

| Model | Parameters incl. training heads | First batch NTP CE | Final batch NTP CE | Official sampled test solve rate | Greedy test solve rate |
|---|---:|---:|---:|---:|---:|
'''
for r in records:
    text+=f'| {r["algorithm"]} | {r["parameters"]:,} | {r["initial_ntp_loss"]:.4f} | {r["final_batch_ntp_loss"]:.4f} | {r["test"]["official_sampled_path_accuracy"]:.2%} | {r["test"]["greedy_path_accuracy"]:.2%} |\n'
text+='''
The NTP losses are individual minibatch measurements, not averaged held-out losses.
The official accuracy metric generates the complete path without teacher forcing,
using temperature-1 sampling. Greedy evaluation is supplementary. No best-test
checkpoint is selected; the final checkpoint at the fixed budget is evaluated.

## What we can conclude

The official code, published task, losses, evaluation, and checkpoints run end to
end. The small pilot does not establish a NextLat advantage or reproduce the
paper's accuracy. Low solve rates at this budget can reflect insufficient training,
model capacity, data size, or implementation/runtime differences; they are not
by themselves evidence against the paper. Full paper-scale validation remains open.

## Next experiment

Use `configs/paper_g5_5.json` on a GPU: 12 layers, width 384, 6 heads, 200,000
training graphs, 20,000 test graphs, batch 512, and 20,000 steps. Start with one
seed per method; then use the paper's five seeds (1234–1238). The paper-scale runner
uses single-GPU bf16 with gradient accumulation instead of the authors' multi-GPU
bf16 launch and uses matched
initial trunk weights and a separate validation set; disclose these differences.
Keep compilation disabled as advised by the official repository. Measure GPU
runtime before committing the complete budget. Full training results are pending.

## Cluster validation and submitted runs

On NYU HPC Cloud Bursting, all eight checks passed under Python 3.12.16 and
PyTorch 2.6.0+cu124. A single A100-SXM4-40GB execution check (job 29555) completed
with Slurm COMPLETED/0:0 in 44 seconds. Each model ran 10 optimizer steps using
the 12-layer, 384-wide architecture, bf16, and effective batch 512. This check
used 1,024 training and 64 test examples; it is not a converged benchmark result.
Both sampled test solve rates were 0/64 at that short budget.

The full corpus is prepared on the cluster. GPT job 29574 is queued, followed
by NextLat job 29575 after successful GPT completion. Both use 20,000 optimizer
steps for seed 1234, one A100, and a ten-hour limit. Smoke step times give rough
screening estimates of 3.32 hours for GPT and 4.04 hours for NextLat, including
25% headroom and five minutes of overhead. Actual full-run durations and accuracy
remain unmeasured. The CPU preparation session was explicitly released.

See cluster/STATUS.md and cluster/validation_29555/ for downloaded raw logs,
environment packages, data hashes, and execution-check results.

Only after this stage should we transfer the verified NextLat model/objective to
our delayed-memory study and interpret differences in recall, probes, and geometry.

## Sources

- Official code: https://github.com/JaydenTeoh/NextLat
- Paper, version 4: https://arxiv.org/abs/2511.05963v4
- Original Path-Star task: https://arxiv.org/abs/2403.06963

See README.md for commands, provenance, and a configuration comparison. Raw numbers,
optimizer checkpoints, and validation logs are in `runs/`. Dataset hashes and sizes
are in `datasets/pilot/manifest.json`.
'''
(ROOT/'RESULTS.md').write_text(text)
paths=['data/stargraph/prepare.py','data/stargraph.py','models/model_gpt.py','models/model_nextlat.py','models/model_base.py','config/stargraph/5_5/gpt_stargraph_5_5.yaml','config/stargraph/5_5/nextlat_stargraph_5_5.yaml','LICENSE']
manifest={'repository':'https://github.com/JaydenTeoh/NextLat','commit':subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip(),'paper':'https://arxiv.org/abs/2511.05963v4','sha256':{p:hashlib.sha256((ROOT/'upstream'/p).read_bytes()).hexdigest() for p in paths}}
(ROOT/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
print('Saved comparison, RESULTS.md, and PROVENANCE.json.')
