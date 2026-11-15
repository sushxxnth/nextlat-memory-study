"""Audit and report the factorial supervision/length controls without test selection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from memory.experiment import ROOT
from memory.sweep_report import summary


def measures(result,delay):
    p=result['test'][str(delay)]
    return dict(recall=p['answer_accuracy'],restricted=p['constrained_answer_accuracy'],
                non_answer=p['non_answer_token_rate'],
                prequery_memory_probe=p['probes']['before_query/stored_bits_mean']['accuracy'],
                atquery_answer_probe=p['probes']['at_query/cue']['accuracy'],
                **({k:p[k] for k in ['latent_one_step_accuracy','latent_rollout_accuracy']} if result['algorithm']=='nextlat' else {}))


def report(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==(root/'plan.sha256').read_text().strip()
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==digest,name
    records={};groups={};families={};rows=[]
    for e in plan['entries']:
        folder=root/e['relative_run'];r=json.loads((folder/'results.json').read_text());c=e['config']
        assert r['status']=='completed' and r['steps']==c['steps'] and r['algorithm']==e['algorithm']
        assert json.loads((folder/'config.json').read_text())==c
        with (folder/'training.csv').open() as stream:history=list(csv.DictReader(stream))
        assert [int(h['step']) for h in history]==list(range(1,c['steps']+1))
        assert all(np.isfinite(float(value)) for h in history for value in h.values())
        for h in history:
            expected=2 if c['loss_scope']=='response' else int(h['delay'])+c['num_cues']+4
            assert int(h['supervised_ntp_tokens_per_example'])==expected
            assert int(h['token_context_length'])==(int(h['delay'])+c['num_cues']+2 if c['loss_scope']=='response' else 0)
        assert r['training_data_digest']
        f=json.loads((folder/'families.json').read_text())
        assert set(f['probe_train'])=={'4','8','16'}
        assert not set(sum(f['probe_train'].values(),[]))&set(sum(f['test'].values(),[]))
        key=e['condition'],r['seed'];records[key]=(e,r);families[key]=f
        groups.setdefault(e['condition'],[]).append((c,r))
        for delay in c['eval_delays']:
            rows.append(dict(condition=e['condition'],seed=r['seed'],delay=delay,loss_scope=c['loss_scope'],
                             trained_delay=delay in c['train_delays'],**measures(r,delay)))
    gates={};aggregate={};training={}
    for condition,parts in groups.items():
        aggregate[condition]={}
        training[condition]=dict(common=summary([min(r['test'][str(d)]['answer_accuracy'] for d in [4,8,16]) for c,r in parts]),
                                 all_trained=summary([min(r['test'][str(d)]['answer_accuracy'] for d in c['train_delays']) for c,r in parts]))
        for d in parts[0][0]['eval_delays']:
            aggregate[condition][str(d)]={k:summary([measures(r,d)[k] for c,r in parts]) for k in measures(parts[0][1],d)}
        gates[condition]={}
        distribution=condition.split('_')[0]
        for c,r in parts:
            seed=r['seed'];control=f'{distribution}_{c["loss_scope"]}_gpt'
            baseline=records[control,seed][1]
            assert r['training_data_digest']==baseline['training_data_digest']
            # Probe/evaluation samples are identical across ALL cells of a seed.
            assert families[condition,seed]==families['sparse_full_gpt',seed]
            # Both masks and both models share the schedule's exact training stream.
            assert r['training_data_digest']==records[f'{distribution}_full_gpt',seed][1]['training_data_digest']
            passed=lambda delays:all(min(r['test'][str(d)]['answer_accuracy'],baseline['test'][str(d)]['answer_accuracy'])>=plan['in_distribution_gate'] for d in delays)
            gates[condition][str(seed)]=dict(common=passed([4,8,16]),all_trained=passed(c['train_delays']))
    contrasts={}
    def contrast(name,positive,negative):
        contrasts[name]={str(d):{metric:summary([measures(records[positive,s][1],d)[metric]-measures(records[negative,s][1],d)[metric] for s in plan['seeds']])
                        for metric in ['recall','restricted','non_answer','prequery_memory_probe']} for d in [12,32,64]}
    for algorithm in ['gpt','nextlat']:
        for distribution in ['sparse','dense']:
            contrast(f'response_minus_full/{distribution}/{algorithm}',f'{distribution}_response_{algorithm}',f'{distribution}_full_{algorithm}')
        for mask in ['full','response']:
            contrast(f'dense_minus_sparse/{mask}/{algorithm}',f'dense_{mask}_{algorithm}',f'sparse_{mask}_{algorithm}')
        contrasts[f'mask_by_distribution_interaction/{algorithm}']={str(d):{metric:summary([
            measures(records[f'dense_response_{algorithm}',s][1],d)[metric]-measures(records[f'dense_full_{algorithm}',s][1],d)[metric]
            -measures(records[f'sparse_response_{algorithm}',s][1],d)[metric]+measures(records[f'sparse_full_{algorithm}',s][1],d)[metric]
            for s in plan['seeds']]) for metric in ['recall','restricted','non_answer']} for d in [12,32,64]}
    for distribution in ['sparse','dense']:
        for mask in ['full','response']:
            contrast(f'nextlat_minus_ntp/{distribution}/{mask}',f'{distribution}_{mask}_nextlat',f'{distribution}_{mask}_gpt')
    output=root/'summary';output.mkdir(exist_ok=True)
    payload=dict(completed_runs=len(records),planned_runs=len(plan['entries']),plan=plan,aggregate=aggregate,
                 minimum_training_recall=training,learning_gates=gates,factor_contrasts=contrasts)
    (output/'aggregate.json').write_text(json.dumps(payload,indent=2)+'\n')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'per_seed.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    def fmt(v):return f'{v["mean"]:.1%} ± {v["sample_sd"]:.1%}' if v['sample_sd'] is not None else f'{v["mean"]:.1%} (one seed)'
    def pp(v):return f'{100*v["mean"]:+.1f} ± {100*v["sample_sd"]:.1f}' if v['sample_sd'] is not None else f'{100*v["mean"]:+.1f} (one seed)'
    lines=['# Four-cue supervision and delay-coverage controls','', '**Aditi & Sushanth — Science of Deep Learning**','',
           f'Completed {len(records)}/{len(plan["entries"])} runs; fresh seeds {plan["seeds"]}; '
           f'{plan["entries"][0]["config"]["steps"]} steps per run. Mean ± sample SD across training seeds; not confidence intervals.','',
           'All cells use a four-cue task, 2-layer/128-wide/4-head transformer, batch 32, AdamW learning rate 0.0005, '
           'weight decay 0.1 and gradient clip 1. Runs execute on CPU in float32. These are small-model controls, not paper-scale replications.','',
           'Sparse training uses delays 4,8,16. Dense training samples every integer 4–16 uniformly. '
           'Response-only supervision uses the authors\' prompt mask on NTP and KL: only ANSWER/EOS targets are supervised. '
           'Context Smooth L1 remains active. Full-sequence supervision matches the previous experiment. '
           'NextLat has weight 1, horizon 3 and predictor width 128 throughout. '
           'Primary recall is always unrestricted full-vocabulary generation; response masking changes training only. '
           'Two-answer recall is a separately labeled diagnostic.','',
           '| Condition | Min common-delay recall | Min all-trained-delay recall | Delay 12 recall | Delay 32 recall | Delay 64 recall | Two-answer recall 64 | Non-answer rate 64 | Full learning gate |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for condition,parts in groups.items():
        g=sum(v['all_trained'] for v in gates[condition].values())
        v=aggregate[condition]
        lines.append(f'| {condition} | {fmt(training[condition]["common"])} | {fmt(training[condition]["all_trained"])} | '
                     f'{fmt(v["12"]["recall"])} | {fmt(v["32"]["recall"])} | {fmt(v["64"]["recall"])} | '
                     f'{fmt(v["64"]["restricted"])} | {fmt(v["64"]["non_answer"])} | {g}/{len(parts)} |')
    lines+=['','## Paired effects at delay 64','',
            'Effects below are paired within seed; positive recall is an increase, while negative non-answer rate is a reduction in wrong token types. '
            'All seeds are retained even if a learning gate fails. Such cases remain confounded by training performance.','',
            '| Contrast | Recall difference (percentage points) | Non-answer rate difference (percentage points) |','|---|---:|---:|']
    for name,values in contrasts.items():
        lines.append(f'| {name} | {pp(values["64"]["recall"])} | {pp(values["64"]["non_answer"])} |')
    lines+=['','## Verification and interpretation limits','',
            'The source snapshot and plan hashes match. All training histories are finite and sequential. '
            'Models/masks share exact training streams within each schedule; all four cells share the same evaluation and probe-training families per seed. '
            'Probe training uses delays 4,8,16 in every cell, avoiding unequal decoder-training budgets. '
            'Probe/test families are disjoint. The response/EOS never enters an evaluation forward pass.','',
            'The full gate requires both matched models to reach at least 95% recall at every delay trained in that cell. '
            'The common gate uses only 4,8,16. Delay 12 is trained in the dense cells and unseen in the sparse cells; '
            '24,32,48,64 are unseen for every cell. Better performance at 12 in dense training is not OOD extrapolation.','',
            'All cells have equal optimization steps and examples, but sparse/dense schedules have slightly different expected token counts. '
            'Response masking changes both NTP and KL target normalization and their balance against context Smooth L1. '
            'It is a token-supervision intervention, not a head-only intervention. Frozen-probe failures do not prove erasure, '
            'and long rollouts can accumulate transition error. These controls do not resolve all positional or memory-capacity confounds.']
    if 'continuation' in plan:
        lines+=['','## Uniform budget extension','',
                f'All 24 cells were continued from {plan["continuation"]["source_budget"]} to '
                f'{plan["entries"][0]["config"]["steps"]} steps after several common-delay validation scores '
                'remained below 95%. Original checkpoints/results are preserved. We retained model weights, '
                'optimizer state, data RNG and training-stream digest. An integration test verified exact equality '
                'between continued and uninterrupted training. This is 24 continued training runs, not 48 independent seeds. '
                'Initial test results had already been visible, so this extension is exploratory. '
                'No best-test checkpoint was selected; intermediate and final results are both retained.']
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    styles=[('gpt','full','#345a7d','-'),('nextlat','full','#b55d38','-'),
            ('gpt','response','#345a7d','--'),('nextlat','response','#b55d38','--')]
    for row,distribution in enumerate(['sparse','dense']):
        for column,metric in enumerate(['recall','non_answer']):
            ax=axes[row,column]
            for algorithm,mask,color,line in styles:
                condition=f'{distribution}_{mask}_{algorithm}';vals=aggregate[condition];ds=sorted(map(int,vals))
                m=np.array([vals[str(d)][metric]['mean'] for d in ds]);s=np.array([vals[str(d)][metric]['sample_sd'] or 0 for d in ds])
                failed=any(not g['all_trained'] for g in gates[condition].values())
                label=f'{"NTP" if algorithm=="gpt" else "NextLat"}, {mask}'+(' *' if failed else '')
                ax.plot(ds,m,linestyle=line,marker='o',ms=2.5,color=color,label=label)
                ax.fill_between(ds,np.maximum(0,m-s),np.minimum(1,m+s),color=color,alpha=.07)
            ax.axvline(16,ls=':',lw=.8,color='gray')
            ax.set(xscale='log',ylim=(-.03,1.03),title=f'{distribution.capitalize()} training: {"answer accuracy" if metric=="recall" else "non-answer token rate"}')
            ax.set_xticks([4,8,12,16,24,32,48,64],[4,8,12,16,24,32,48,64]);ax.legend(frameon=False,fontsize=7)
            ax.set_ylabel('Fraction')
    for ax in axes[1]:ax.set_xlabel('Distractor tokens')
    fig.suptitle('Four-cue controls; bands show ±1 sample SD across seeds',fontsize=12)
    fig.text(.5,.012,'* At least one seed fails the matched learning gate. Vertical line: maximum trained delay (16).',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    lines+=['','![Factorial control comparison](comparison.png)']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(output/'RESULTS.md',flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    report(p.parse_args().run)
