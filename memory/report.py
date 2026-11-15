"""Summarize completed paired memory pilots, without implying multi-seed evidence."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def readout_audit(root, records):
    import torch
    from memory.task import ANSWER_RED, ANSWER_BLUE
    audit={}
    for name, record in records.items():
        state=torch.load(root/name/'checkpoint.pt',map_location='cpu',weights_only=False)
        features=np.load(root/name/'test_features.npz')
        audit[name]={}
        for delay in record['test']:
            h=torch.from_numpy(features[f'delay{delay}/at_query'])
            logits=h @ state['model']['lm_head.weight'].T
            predicted=logits.argmax(-1).numpy();cue=features[f'delay{delay}/cue']
            audit[name][delay]=dict(
                predicted_token_counts_by_cue={str(c):{str(t):int(sum(predicted[cue==c]==t))
                    for t in np.unique(predicted[cue==c])} for c in [0,1]},
                wrong_token_family_rate=float(np.mean((predicted!=ANSWER_RED)&(predicted!=ANSWER_BLUE))))
    (root/'readout_audit.json').write_text(json.dumps(audit,indent=2))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run
    records={name:json.loads((root/name/'results.json').read_text()) for name in ['gpt','nextlat']}
    configs={name:json.loads((root/name/'config.json').read_text()) for name in records}
    assert configs['gpt']==configs['nextlat']
    assert all(r['status']=='completed' for r in records.values())
    # Align counterfactual evaluation families across methods before comparing.
    families={name:json.loads((root/name/'families.json').read_text()) for name in records}
    assert families['gpt']==families['nextlat']
    c=configs['gpt'];delays=c['eval_delays']
    audit=readout_audit(root,records)
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    colors={'gpt':'#254d74','nextlat':'#a8502c'}
    for name,r in records.items():
        data=[r['test'][str(d)] for d in delays];label='NTP' if name=='gpt' else 'NextLat'
        axes[0,0].errorbar(delays,[p['answer_accuracy'] for p in data],
                          yerr=[p['answer_accuracy_family_se'] for p in data],fmt='o-',label=label,color=colors[name],capsize=2)
        for site,style in [('before_query','-'),('at_query','--')]:
            axes[0,1].plot(delays,[p['probes'][f'{site}/cue']['accuracy'] for p in data],
                           marker='o',linestyle=style,label=f'{label}, {site.replace("_"," ")}',color=colors[name])
        axes[1,0].plot(delays,[p['probes']['before_query/nuisance']['accuracy'] for p in data],
                       'o-',label=label,color=colors[name])
    data=[records['nextlat']['test'][str(d)] for d in delays]
    axes[1,1].plot(delays,[p['latent_one_step_accuracy'] for p in data],'o-',label='One dynamics step')
    axes[1,1].plot(delays,[p['latent_rollout_accuracy'] for p in data],'s--',label='Long dynamics rollout')
    titles=['Full-attention answer accuracy','Cue probe accuracy (single hidden state)',
            'Nuisance probe accuracy before query','NextLat output-head accuracy without attention']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_ylim(-.03,1.03)
        ax.axvspan(min(c['train_delays']),max(c['train_delays']),color='gray',alpha=.08)
        ax.axhline(.25 if ax is axes[1,0] else .5,color='gray',linewidth=.7,linestyle=':')
        ax.set_xscale('log',base=2);ax.set_xticks(delays,[str(d) for d in delays]);ax.legend(frameon=False,fontsize=8)
    for ax in axes[1]:ax.set_xlabel('Distractor tokens (delay)')
    fig.suptitle(f'Delayed memory: {c["steps"]} steps, seed {c["seed"]}; shaded training-delay range',fontsize=12)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    lines=['# Controlled memory pilot', '', '**Aditi & Sushanth — Science of Deep Learning**', '',
           f'Completed paired pilot: {c["layers"]} layers, width {c["width"]}, {c["heads"]} heads, '
           f'batch {c["batch_size"]}, {c["steps"]} steps, seed {c["seed"]}. '
           f'Training delays: {c["train_delays"]}. NextLat auxiliary weight {c["lambda_aux"]} '
           f'for each of Smooth L1 and KL; horizon {c["horizon"]}; predictor hidden width '
           f'{records["nextlat"]["predictor_hidden_width"]}.', '',
           'The official models and losses are unchanged. The sequence task and analysis are our extension. '
           'Both methods share initial transformer weights and training/evaluation histories. '
           'The final fixed-budget checkpoint is evaluated, without test checkpoint selection.', '',
           '| Model | Delay | Full-vocabulary recall | Two-answer recall | Cue probe before query | Cue probe at query | Nuisance probe before query |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in records.items():
        for delay in delays:
            p=r['test'][str(delay)]
            lines.append(f'| {name} | {delay} | {p["answer_accuracy"]:.1%} | {p["constrained_answer_accuracy"]:.1%} | '
                         f'{p["probes"]["before_query/cue"]["accuracy"]:.1%} | '
                         f'{p["probes"]["at_query/cue"]["accuracy"]:.1%} | '
                         f'{p["probes"]["before_query/nuisance"]["accuracy"]:.1%} |')
    lines+=['','Full-vocabulary answer recall is reported. Cue chance is 50%, nuisance chance 25%. '
            'The accuracy plot uses standard errors across noise families; these are not uncertainty across training seeds.', '',
            '| Delay | NextLat one-step recall | NextLat long-rollout recall |', '|---:|---:|---:|']
    for delay in delays:
        p=records['nextlat']['test'][str(delay)]
        lines.append(f'| {delay} | {p["latent_one_step_accuracy"]:.1%} | {p["latent_rollout_accuracy"]:.1%} |')
    longest=str(max(delays));p=records['nextlat']['test'][longest]
    lines+=['', '## Initial diagnostic observation', '',
            f'At delay {longest}, NextLat has {p["answer_accuracy"]:.1%} full-vocabulary recall, '
            f'{p["constrained_answer_accuracy"]:.1%} accuracy when choosing only between the two answer tokens, '
            f'and {p["probes"]["at_query/cue"]["accuracy"]:.1%} cue-probe accuracy at QUERY. '
            f'{audit["nextlat"][longest]["wrong_token_family_rate"]:.1%} of its predictions are outside the answer-token family. '
            'The saved-state/readout audit uses the existing head without fitting or retraining. '
            'The cue can remain linearly accessible even when unconstrained output prediction fails; '
            'this alone does not establish memory erasure. See readout_audit.json for prediction counts.', '',
            '## Interpretation limits', '',
            'This is one training seed and a short training budget. Establish reliable in-distribution '
            'learning, then compare multiple matched seeds and the auxiliary-weight/predictor-capacity sweeps. '
            'There is no requirement that compression cause a memory penalty.', '',
            'A linear probe measures accessibility to that classifier; chance performance does not prove '
            'all information was erased. Cosine similarity is descriptive. A weak long latent rollout '
            'can reflect accumulated predictor error, especially beyond the training horizon. '
            'Use full-attention recall, query versus pre-query probes, and one-step/long-rollout '
            'controls together before claiming harmful aliasing.', '',
            'Geometry, teacher-state MSE, family-level errors, raw features, frozen probes, source commit, '
            'configuration, logs, and optimizer checkpoints are saved in each model directory.', '',
            '![Memory pilot comparison](comparison.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(root/'REPORT.md');print(root/'comparison.png')


if __name__=='__main__':main()
