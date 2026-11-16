"""Report all completed budgets/seeds, keeping readout and memory metrics separate."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    rows=[]
    for step in plan['stages']:
        for entry in plan['entries']:
            folder=root/f'step_{step}'/entry['relative_run']
            if not (folder/'readout_refit.json').exists():continue
            result=json.loads((folder/'results.json').read_text())
            refit=json.loads((folder/'readout_refit.json').read_text())
            gate=all(result['test'][str(d)]['answer_accuracy']>=.95 for d in entry['config']['train_delays'])
            for d, part in result['test'].items():
                head=refit['test'][d]
                rows.append(dict(condition=entry['condition'],seed=entry['config']['seed'],step=step,
                    delay=int(d),learning_gate=gate,native=part['answer_accuracy'],
                    non_answer=part['non_answer_token_rate'],restricted=part['constrained_answer_accuracy'],
                    native_head_redecode=head['native_cpu_redecode']['accuracy'],
                    refitted=head['refitted']['accuracy'],refitted_non_answer=head['refitted']['non_answer_rate'],
                    head_only_gain=head['refitted']['accuracy']-head['native_cpu_redecode']['accuracy'],
                    before_query_bits=part['probes']['before_query/stored_bits_mean']['accuracy'],
                    at_query_answer_probe=part['probes']['at_query/cue']['accuracy'],
                    refit_gradient_max=refit['optimizer']['gradient_max']))
    if not rows:raise ValueError('No completed model + readout analyses yet.')
    out=root/'summary';out.mkdir(exist_ok=True)
    with (out/'per_seed.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    records=[]
    for step in plan['stages']:
        for condition in dict.fromkeys(e['condition'] for e in plan['entries']):
            for delay in plan['entries'][0]['config']['eval_delays']:
                group=[r for r in rows if r['step']==step and r['condition']==condition and r['delay']==delay]
                if not group:continue
                item=dict(step=step,condition=condition,delay=delay,seeds=[r['seed'] for r in group],
                          completed_seeds=len(group),planned_seeds=len(plan['seeds']),
                          passed_learning_gate=sum(r['learning_gate'] for r in group))
                for key in list(rows[0])[5:]:
                    values=[r[key] for r in group]
                    item[key]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
                records.append(item)
    (out/'aggregate.json').write_text(json.dumps(records,indent=2)+'\n')
    # Training streams must match within seed and budget across supervision and weights.
    streams=[]
    for step in plan['stages']:
        for seed in plan['seeds']:
            digests=[]
            for e in plan['entries']:
                p=root/f'step_{step}'/e['relative_run']/'results.json'
                if e['config']['seed']==seed and p.exists():digests.append(json.loads(p.read_text())['training_data_digest'])
            if digests and len(set(digests))!=1:raise RuntimeError('Training streams differ across matched cells.')
            streams.append(dict(step=step,seed=seed,completed_models=len(digests),matched=len(set(digests))<=1))
    (out/'stream_audit.json').write_text(json.dumps(streams,indent=2)+'\n')
    def pct(item,key):
        v=item[key];return f"{100*v['mean']:.1f}"+(f" +/- {100*v['sd']:.1f}" if v['sd'] is not None else ' (one seed)')
    lines=['# GPU supervision and readout progression',
        'Exploratory: fixed budgets and endpoints; unchanged noise-family partitions can overlap earlier evaluations. Mean +/- sample SD across completed training seeds; incomplete cells are explicitly labeled.',
        '| Budget | Condition | Seeds | Learning gate | Native recall | Non-answer rate | Native frozen head | Refitted frozen head | Pre-query cue probe |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in records:
        if item['delay']==64:
            lines.append(f"| {item['step']} | {item['condition']} | {item['completed_seeds']}/{item['planned_seeds']} | {item['passed_learning_gate']}/{item['completed_seeds']} | {pct(item,'native')} | {pct(item,'non_answer')} | {pct(item,'native_head_redecode')} | {pct(item,'refitted')} | {pct(item,'before_query_bits')} |")
    lines+=['','All values in the table are percentages at delay 64. Gate: >=95% native recall at every trained delay; no seed is discarded. Early budgets may underfit.',
        'Refitting changes only the native bias-free 24-class output matrix, using separate probe families. Both frozen-head columns decode identical cached at-query states in CPU float64; native GPU recall is listed separately. No vocabulary restriction is applied.',
        'A head-only improvement is evidence of accessible answer information after the query and attention. It does not establish pre-query memory, and poor refit/probe performance does not prove information erasure. The pre-query probe decodes all four stored bits before the query index arrives.',
        'Supervision masks also change loss normalization and representation training. Lambda varies both latent Smooth L1 and KL together, so this is not a decomposition of their separate effects. Native-head refits have a fixed optimizer cap; inspect per-run gradient diagnostics.',
        'A single-seed or partially completed table is a progress snapshot, not the final five-seed result.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    final=max(r['step'] for r in records)
    for condition in dict.fromkeys(e['condition'] for e in plan['entries']):
        rs=sorted([r for r in records if r['step']==final and r['condition']==condition],key=lambda r:r['delay'])
        if not rs:continue
        style='--' if condition.startswith('full_') else '-'
        for ax,key in zip(axes,['native','refitted','before_query_bits']):
            ax.plot([r['delay'] for r in rs],[100*r[key]['mean'] for r in rs],style,label=f"{condition} (n={rs[0]['completed_seeds']}/{rs[0]['planned_seeds']})")
    for ax,title in zip(axes,['Native unrestricted recall','Frozen head after refit','Before-query cue decoding']):
        ax.set(title=f'{title} (budget {final})',xlabel='Distractor tokens',ylabel='Accuracy (%)',ylim=(0,105))
        ax.axvspan(4,16,alpha=.05,color='grey');ax.grid(alpha=.2)
    axes[2].legend(fontsize=6,loc='lower left')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(out/'RESULTS.md',flush=True)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    report(p.parse_args().run)
