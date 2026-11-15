"""Audit matched data streams and summarize final-budget controlled runs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def summary(values):
    return dict(mean=float(np.mean(values)), sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                values=[float(v) for v in values])


def measures(result, delay):
    part=result['test'][str(delay)]
    cue_key='before_query/cue' if result['num_cues']==1 else 'before_query/stored_bits_mean'
    return dict(recall=part['answer_accuracy'], restricted=part['constrained_answer_accuracy'],
                prequery_memory_probe=part['probes'][cue_key]['accuracy'],
                atquery_answer_probe=part['probes']['at_query/cue']['accuracy'],
                nuisance_probe=part['probes']['before_query/nuisance']['accuracy'],
                prequery_rank=part['geometry']['before_query']['covariance_entropy_effective_rank'],
                **({k:part[k] for k in ['latent_one_step_accuracy','latent_rollout_accuracy']} if result['algorithm']=='nextlat' else {}))


def report(root):
    root=Path(root)
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==(root/'plan.sha256').read_text().strip()
    plan=json.loads((root/'plan.json').read_text())
    loaded={}; rows=[]; pending=[]; groups={}
    for entry in plan['entries']:
        folder=root/entry['relative_run']
        if not (folder/'results.json').exists():pending.append(entry['relative_run']);continue
        result=json.loads((folder/'results.json').read_text())
        assert result['status']=='completed' and result['steps']==entry['config']['steps']
        assert result['algorithm']==entry['algorithm']
        assert json.loads((folder/'config.json').read_text())==entry['config']
        with (folder/'training.csv').open() as stream:history=list(csv.DictReader(stream))
        assert [int(row['step']) for row in history]==list(range(1,result['steps']+1))
        assert all(np.isfinite(float(v)) for row in history for v in row.values())
        assert result['training_data_digest']
        family=json.loads((folder/'families.json').read_text())
        train=set(sum(family['probe_train'].values(),[]));test=set(sum(family['test'].values(),[]))
        assert not train&test
        key=entry['condition'],result['seed']
        loaded[key]=(entry,result,family)
        groups.setdefault(entry['condition'],[]).append(result)
        for delay in entry['config']['eval_delays']:
            rows.append(dict(condition=entry['condition'],algorithm=result['algorithm'],seed=result['seed'],
                             num_cues=result['num_cues'],lambda_aux=entry['config']['lambda_aux'],
                             predictor_width=result['predictor_hidden_width'],delay=delay,
                             **measures(result,delay)))
    # Every condition at the same memory load/seed must consume exactly the same
    # training stream and evaluation families as its matched NTP control.
    gates={};paired={}
    for (condition, seed),(entry,result,family) in loaded.items():
        control='ntp' if result['num_cues']==1 else f'cues{result["num_cues"]}_gpt'
        if (control,seed) not in loaded:continue
        baseline=loaded[control,seed][1]
        assert family==loaded[control,seed][2],(condition,seed,'evaluation mismatch')
        assert result['training_data_digest']==baseline['training_data_digest'],(condition,seed,'training mismatch')
        delays=entry['config']['train_delays']
        gate=all(min(result['test'][str(d)]['answer_accuracy'],baseline['test'][str(d)]['answer_accuracy'])>=plan['in_distribution_gate'] for d in delays)
        gates[f'{condition}/seed{seed}']=gate
        paired.setdefault(condition,[]).append(dict(seed=seed,gate=gate,
            difference_at_delay64=result['test']['64']['answer_accuracy']-baseline['test']['64']['answer_accuracy']))
    aggregate={}
    for condition,results in groups.items():
        delays=sorted(map(int,results[0]['test']))
        aggregate[condition]={str(d):{metric:summary([measures(r,d)[metric] for r in results])
                              for metric in measures(results[0],d)} for d in delays}
    id_minimum={condition:summary([min(r['test'][str(d)]['answer_accuracy'] for d in [4,8,16]) for r in results])
                for condition,results in groups.items()}
    payload=dict(planned_runs=len(plan['entries']),completed_runs=len(loaded),pending=pending,id_minimum=id_minimum,
                 configuration=plan,aggregate=aggregate,paired=paired,matched_id_gates=gates)
    target=root/'summary';target.mkdir(exist_ok=True)
    (target/'aggregate.json').write_text(json.dumps(payload,indent=2)+'\n')
    if rows:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with (target/'per_seed.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def cell(condition,delay,metric='recall'):
        v=aggregate[condition][str(delay)][metric]
        return f'{v["mean"]:.1%} ± {v["sample_sd"]:.1%}' if v['sample_sd'] is not None else f'{v["mean"]:.1%} (one seed)'
    lines=['# Controlled memory study: preliminary results','', '**Aditi & Sushanth — Science of Deep Learning**','',
           f'Completed {len(loaded)} of {len(plan["entries"])} planned runs. Seeds: {plan["seeds"]}. '
           f'Training budget: {plan["entries"][0]["config"]["steps"]} steps per run. '
           'Entries show mean ± sample standard deviation across training seeds, not confidence intervals.','',
           'The official GPT/NextLat models and losses remain unchanged. Our task, sweep, probes and reporting are the extension. '
           'Within each cue count and seed, matched training-stream digests and evaluation families are verified. '
           'All saved training rows are finite and sequential; probe and test families are disjoint.','',
           '| Condition | Cues | Seeds | Minimum trained-delay recall | Recall delay 12 (unseen) | Recall delay 32 | Recall delay 64 | Restricted recall 64 | Pre-query memory probe 64 | Matched ID gate passes |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for condition,results in groups.items():
        gate_count=sum(p['gate'] for p in paired.get(condition,[]))
        v=id_minimum[condition]
        minimum=f'{v["mean"]:.1%} ± {v["sample_sd"]:.1%}' if v['sample_sd'] is not None else f'{v["mean"]:.1%} (one seed)'
        lines.append(f'| {condition} | {results[0]["num_cues"]} | {len(results)} | {minimum} | {cell(condition,12)} | {cell(condition,32)} | {cell(condition,64)} | {cell(condition,64,"restricted")} | {cell(condition,64,"prequery_memory_probe")} | {gate_count}/{len(results)} |')
    lines+=['','## Interpretation rules','',
            'An ID gate passes only when both a condition and its matched NTP control achieve at least 95% recall at every trained delay (4, 8, 16). '
            'A failed gate means that generalization differences are confounded by in-distribution learning. '
            'All conditions are reported, regardless of the gate or outcome.','',
            'The multiple-cue task stores independent bits, then reveals the queried index only after the distractors. '
            'For more than one cue, the pre-query memory probe is the mean accuracy of separate linear decoders for all stored bits. '
            'It does not assume the model knows the future query index. A separately labeled selected_bit_given_query metric '
            'uses that index to choose among decoders; this diagnostic has query side information. '
            'At-query cue probes predict the requested answer. Binary probe chance is 50%; nuisance chance is 25%.','',
            'The paired counterfactual groups contain both values of the queried bit and all four nuisance tags with identical distractors, '
            'other bits and query index. Other stored bits vary across families. Predictor widths 128/256/512 change the auxiliary MLP capacity, '
            'not the full transformer hidden-state dimension or causal-attention access.','',
            'Low linear-probe accuracy alone does not prove erasure; latent rollouts may accumulate dynamics error. '
            'Compare unrestricted versus two-answer accuracy to detect readout/token-type failures. '
            'The one-factor design does not estimate loss-weight × width interactions. '
            'The longer training budget is a follow-up after inspecting the earlier pilot, not an independent confirmation on a new task distribution.','',
            'Only delays 4, 8 and 16 are trained: delay 12 is an unseen interpolation test. Failures at 12 '
            'as well as longer delays can reflect length/position-specific strategies. Delay shift also tests '
            'positional generalization. Across cue counts, prefix length and query vocabulary change; '
            'comparisons within each task are matched, but the cross-task trend does not isolate a pure memory-capacity limit.','',
            'The fixed training budget is an initial follow-up. If ID gates fail or validation is still improving, '
            'a predeclared longer-budget follow-up is needed before interpreting capacity or delay effects.']
    if pending:lines+=['','Pending runs:']+[f'- {p}' for p in pending]
    if len(groups)>0:
        os.environ.setdefault('MPLCONFIGDIR',str(target/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        def draw(ax,names,metric,title):
            for condition in names:
                if condition not in aggregate:continue
                ds=sorted(map(int,aggregate[condition]))
                means=np.array([aggregate[condition][str(d)][metric]['mean'] for d in ds])
                sd=np.array([aggregate[condition][str(d)][metric]['sample_sd'] or 0 for d in ds])
                names={'ntp':'NTP','lambda01':'NextLat λ=0.1','lambda1':'NextLat λ=1 (width 128)',
                       'lambda10':'NextLat λ=10','width256':'NextLat width 256','width512':'NextLat width 512',
                       **{f'cues{k}_{a}':f'{k} cues: {"NTP" if a=="gpt" else "NextLat"}' for k in [4,8,16] for a in ['gpt','nextlat']}}
                failed=any(not x['gate'] for x in paired.get(condition,[]))
                ax.plot(ds,means,'o-',ms=3,label=names[condition]+(' *' if failed else ''))
                ax.fill_between(ds,np.maximum(0,means-sd),np.minimum(1,means+sd),alpha=.1)
            ax.set(xscale='log',ylim=(-.03,1.03),title=title,xlabel='Distractor tokens')
            ax.set_xticks([4,8,12,16,24,32,48,64],[4,8,12,16,24,32,48,64]);ax.tick_params(axis='x',labelsize=8)
            if ax.lines:ax.legend(frameon=False,fontsize=7)
        draw(axes[0],['ntp','lambda01','lambda1','lambda10'],'recall','Auxiliary-weight sweep')
        draw(axes[1],['ntp','lambda1','width256','width512'],'recall','Predictor-width sweep')
        draw(axes[2],['cues4_gpt','cues4_nextlat','cues8_gpt','cues8_nextlat','cues16_gpt','cues16_nextlat'],'recall','Multiple-cue recall')
        axes[0].set_ylabel('Full-vocabulary answer accuracy')
        fig.suptitle('Controlled memory study; bands show ±1 sample SD across seeds',fontsize=11)
        fig.text(.5,.012,'Training delays: 4, 8, 16; delay 12 is unseen.  * At least one matched seed fails the training-delay gate.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(target/'comparison.png',dpi=180);plt.close(fig)
        lines+=['','![Controlled-study comparison](comparison.png)']
    (target/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(target/'RESULTS.md',flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    report(parser.parse_args().run)
