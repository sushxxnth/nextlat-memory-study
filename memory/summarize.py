"""Aggregate matched seeds; report variation across training seeds explicitly."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

METRICS={'answer_accuracy':'Full-vocabulary recall',
         'constrained_answer_accuracy':'Two-answer recall',
         'before_query/cue':'Cue probe before query',
         'at_query/cue':'Cue probe at query'}


def value(part,key):
    return part['probes'][key]['accuracy'] if '/' in key else part[key]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records={};configs=[]
    for run in args.runs:
        for algorithm in ['gpt','nextlat']:
            r=json.loads((run/algorithm/'results.json').read_text());assert r['status']=='completed'
            records[r['seed'],algorithm]=r
            c=json.loads((run/algorithm/'config.json').read_text());c.pop('seed');configs.append(c)
        assert json.loads((run/'gpt/families.json').read_text())==json.loads((run/'nextlat/families.json').read_text())
    assert all(c==configs[0] for c in configs)
    seeds=sorted({s for s,a in records});assert len(seeds)==len(args.runs)
    c=configs[0];delays=c['eval_delays'];n=len(seeds)
    rows=[];aggregate={};deltas={}
    for algorithm in ['gpt','nextlat']:
        aggregate[algorithm]={}
        for delay in delays:
            parts=[records[s,algorithm]['test'][str(delay)] for s in seeds]
            metrics={key:[value(p,key) for p in parts] for key in METRICS}
            aggregate[algorithm][str(delay)]={key:dict(mean=float(np.mean(values)),
                sample_sd=float(np.std(values,ddof=1)) if n>1 else 0.,values=values) for key,values in metrics.items()}
            for seed,part in zip(seeds,parts):rows.append(dict(seed=seed,algorithm=algorithm,delay=delay,
                **{key:value(part,key) for key in METRICS}))
    for delay in delays:
        deltas[str(delay)]={}
        for key in METRICS:
            values=np.array(aggregate['nextlat'][str(delay)][key]['values'])-np.array(aggregate['gpt'][str(delay)][key]['values'])
            deltas[str(delay)][key]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)) if n>1 else 0.,values=values.tolist())
    (args.out/'aggregate.json').write_text(json.dumps(dict(seeds=seeds,config=c,aggregate=aggregate,paired_differences=deltas),indent=2))
    with (args.out/'per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(args.out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,6.4),sharex=True)
    for ax,(key,title) in zip(axes.flat,METRICS.items()):
        for algorithm,color in [('gpt','#254d74'),('nextlat','#a8502c')]:
            data=[aggregate[algorithm][str(d)][key] for d in delays]
            means=np.array([p['mean'] for p in data]);sd=np.array([p['sample_sd'] for p in data])
            ax.plot(delays,means,'o-',color=color,label='NTP' if algorithm=='gpt' else 'NextLat')
            ax.fill_between(delays,np.maximum(means-sd,0),np.minimum(means+sd,1),color=color,alpha=.12)
        ax.axvspan(min(c['train_delays']),max(c['train_delays']),color='gray',alpha=.07)
        ax.axhline(.5,ls=':',color='gray',lw=.7);ax.set_ylim(-.03,1.03)
        ax.set_title(title,fontsize=10);ax.set_xscale('log',base=2)
        ax.set_xticks(delays,[str(d) for d in delays]);ax.legend(frameon=False,fontsize=8)
    for ax in axes[1]:ax.set_xlabel('Distractor tokens (delay)')
    fig.suptitle(f'Memory pilot: {n} matched seeds, {c["steps"]} steps; bands are ±1 sample SD',fontsize=11)
    fig.tight_layout();fig.savefig(args.out/'comparison.png',dpi=180);plt.close(fig)
    lines=['# Initial controlled-memory results','', '**Aditi & Sushanth — Science of Deep Learning**','',
           f'Completed {n} paired CPU seeds: {seeds}. Each model has {c["layers"]} layers, width {c["width"]}, '
           f'{c["steps"]} optimization steps and batch {c["batch_size"]}. Training delays are {c["train_delays"]}. '
           f'NextLat uses lambda={c["lambda_aux"]}, horizon {c["horizon"]}, projection factor {c["proj_factor"]}.','',
           'These are our new memory experiments using the pinned official models. The long cluster '
           'Path-Star runs are separate. Both models share initialization and data within each seed. '
           'Tests use held-out noise families; probes train on their own disjoint families.','',
           'Entries are mean ± sample standard deviation across training seeds, not confidence intervals. '
           'Full-vocabulary and restricted two-answer accuracy are kept separate to expose token-type/readout failures.','',
           '| Model | Delay | Full-vocabulary recall | Two-answer recall | Cue probe before query | Cue probe at query |',
           '|---|---:|---:|---:|---:|---:|']
    for algorithm in ['gpt','nextlat']:
        for delay in delays:
            values=[aggregate[algorithm][str(delay)][key] for key in METRICS]
            cells=[f'{v["mean"]:.1%} ± {v["sample_sd"]:.1%}' for v in values]
            lines.append(f'| {algorithm} | {delay} | '+' | '.join(cells)+' |')
    lines+=['','## Interpretation and next experiments','',
            'Compare pre-query accessibility with full-attention recall; successful output can rely '
            'on retrieving the original cue. Compare restricted and unrestricted outputs before '
            'interpreting long-delay errors as forgetting. A linear probe is a limited decoder, '
            'and poor latent rollouts may reflect dynamics drift. These short runs establish a '
            'working experimental method; they do not establish a general compression–memory tradeoff.','',
            'Next: sweep auxiliary weight (0.1, 1, 10) at fixed predictor width and matched seeds, '
            'then predictor hidden width (128, 256, 512) at fixed weight. Reuse the NTP seed controls. '
            'Check longer training and intermediate delays around each transition before making '
            'claims about capacity or memory. Add a state-combination environment after this first study.','',
            'Raw per-seed metrics and paired differences are in aggregate.json/per_seed.csv. '
            'Each run saves frozen probes, hidden states, geometry, latent rollouts and the final checkpoint.','',
            '![Three-seed comparison](comparison.png)','']
    (args.out/'RESULTS.md').write_text('\n'.join(lines))
    print(args.out/'RESULTS.md')


if __name__=='__main__':main()
