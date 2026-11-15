"""Aggregate all published-probe results, with seed-level uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summary(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), values=list(map(float, values)))


def formatted(stat):
    return f"{100*stat['mean']:.1f} +/- {100*stat['sd']:.1f}"


def generate(out):
    out = Path(out)
    records = json.loads((out/'all_results.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    assert len(records) == 24
    root = Path(protocol['source_run'])
    plan = json.loads((root/'plan.json').read_text())
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest() == protocol['source_plan_sha256']
    for record in records:
        assert record['cached_final_states_bitwise_equal']
        assert hashlib.sha256((root/f"{record['condition']}/seed{record['seed']}/{record['algorithm']}/checkpoint.pt").read_bytes()).hexdigest() == record['checkpoint_sha256']
    conditions = sorted(set(r['condition'] for r in records))
    aggregate = {}
    for condition in conditions:
        rs = sorted([r for r in records if r['condition'] == condition], key=lambda r:r['seed'])
        assert [r['seed'] for r in rs] == [1237, 1238, 1239]
        row = dict(gates=sum(r['full_training_delay_gate'] for r in rs), sites={})
        for site in protocol['sites']:
            row['sites'][site] = {}
            for d in map(str, protocol['delays']):
                row['sites'][site][d] = {}
                for method in rs[0]['sites'][site][d]:
                    keys = [k for k in rs[0]['sites'][site][d][method] if not isinstance(rs[0]['sites'][site][d][method][k], list)]
                    row['sites'][site][d][method] = {k: summary([r['sites'][site][d][method][k] for r in rs]) for k in keys}
        row['native_recall'] = {d: summary([r['native_recall'][d] for r in rs]) for d in rs[0]['native_recall']}
        row['existing_logistic_probe'] = {d: summary([r['existing_logistic_probe'][d] for r in rs]) for d in rs[0]['native_recall']}
        aggregate[condition] = row
    (out/'aggregate.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    with (out/'per_seed.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['condition','seed','site','delay','method','bit_accuracy','shuffled_label_accuracy_mean','training_delay_gate'])
        for r in records:
            for site, ds in r['sites'].items():
                for d, methods in ds.items():
                    for method, metrics in methods.items():
                        writer.writerow([r['condition'], r['seed'], site, d, method, metrics['bit_accuracy'], metrics.get('shuffled_training_label_accuracy_mean',''), r['full_training_delay_gate']])
    ce_audits = [r['ce_fit_audit'] for r in records]
    audit = dict(completed_models=24, training_seeds=[1237,1238,1239], frozen_backbone=True,
        trained_on_separate_probe_families=True, train_delays=[4,8,16], examples_per_training_delay=1024,
        test_examples_per_delay=1024, test_delays=protocol['delays'], layers_and_embedding_reported=True,
        cached_final_state_checks=24*11, cached_final_states_bitwise_equal=True,
        original_checkpoints_unchanged=True, source_plan_unchanged=True,
        null_permutations_per_model_site=20, new_probe_unit_tests_passed=3,
        ce_optimizer_max_iterations=200, ce_gradient_max_range=[min(x['gradient_max'] for x in ce_audits), max(x['gradient_max'] for x in ce_audits)],
        ce_optimizer_caveat='Fixed iteration cap; not all fits reached requested gradient tolerance. OLS and ridge provide optimizer-independent checks.',
        reuse='Previously viewed test families and 6000-step checkpoints; exploratory, not an independent replication.')
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig, axes = plt.subplots(2,2,figsize=(10.2,7.2),layout='constrained')
    colors = {'gpt':'#245B80','nextlat':'#A64E24'}
    delays = protocol['delays']
    for algorithm, label in [('gpt','NTP'),('nextlat','NextLat')]:
        a = aggregate[f'dense_response_{algorithm}']; color=colors[algorithm]
        values = [a['sites']['final_norm'][str(d)]['affine_ols']['bit_accuracy'] for d in delays]
        mean=np.array([v['mean']*100 for v in values]);sd=np.array([v['sd']*100 for v in values])
        axes[0,0].plot(delays,mean,'o-',color=color,label=label)
        axes[0,0].fill_between(delays,mean-sd,mean+sd,color=color,alpha=.12)
        layers=['embedding','block1','block2','concat_blocks','final_norm']
        values=[a['sites'][s]['64']['affine_ols']['bit_accuracy'] for s in layers]
        axes[0,1].errorbar(range(5),[100*v['mean'] for v in values],yerr=[100*v['sd'] for v in values],fmt='o-',capsize=3,color=color,label=label)
        methods=['affine_ols','ridge_alpha1','official_ce']
        values=[a['sites']['final_norm']['64'][m]['bit_accuracy'] for m in methods]
        pos=np.arange(3)+(-.16 if algorithm=='gpt' else .16)
        axes[1,0].bar(pos,[100*v['mean'] for v in values],yerr=[100*v['sd'] for v in values],width=.3,capsize=3,color=color,label=label)
        n=a['sites']['final_norm']['16']['affine_ols']['nuisance_accuracy']
        b=a['sites']['final_norm']['16']['affine_ols']['bit_accuracy']
        axes[1,1].bar(np.array([0,1])+(-.16 if algorithm=='gpt' else .16),[100*b['mean'],100*n['mean']],yerr=[100*b['sd'],100*n['sd']],width=.3,capsize=3,color=color,label=label)
    axes[0,0].axvspan(4,16,color='grey',alpha=.08)
    axes[0,0].set(title='A  Cue decoding across retention delays',xlabel='Distractor tokens',ylabel='Four-bit mean accuracy (%)',ylim=(35,105))
    axes[0,1].set(title='B  Location of accessible memory at delay 64',ylabel='Affine cue decoding (%)',ylim=(35,105))
    axes[0,1].set_xticks(range(5),['Token only','Block 1','Block 2','Both blocks','Final norm'],rotation=22)
    axes[1,0].set(title='C  Different probes agree at delay 64',ylabel='Four-bit mean accuracy (%)',ylim=(0,105))
    axes[1,0].set_xticks(range(3),['Affine OLS','Ridge (alpha=1)','Official CE'])
    axes[1,1].set(title='D  Relevant and irrelevant history at delay 16',ylabel='Decoding accuracy (%)',ylim=(0,105))
    axes[1,1].set_xticks([0,1],['Stored cues\n(chance 50%)','Nuisance tag\n(chance 25%)'])
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.axhline(50,color='grey',linestyle=':',linewidth=1)
    axes[0,0].legend(frameon=False,loc='lower left')
    fig.savefig(out/'probe_results.png',dpi=200)
    fig.savefig(out/'probe_results.svg')
    plt.close(fig)
    text = ['# When Does Compression Forget Too Much?',
      '## Midterm check-in: published probes reveal accessible memory in NextLat',
      '**Aditi & Sushanth | CSCI-GA.3033 The Science of Deep Learning | November 14, 2026**',
      'This is an exploratory analysis of 24 existing, fixed-final-budget models, not 24 new training runs. All uncertainty below is sample standard deviation across three independent training seeds. Counterfactual examples and 20 shuffled fits are not treated as independent training seeds.',
      '### Primary result at delay 64',
      'Dense training uses every delay 4-16; response supervision scores ANSWER/EOS for NTP/KL while retaining context latent Smooth L1. Probes see a single state before the query index arrives and decode all four stored bits independently. Binary chance is 50%.',
      '| Probe | NTP (%) | NextLat (%) |','|---|---:|---:|']
    for m,label in [('affine_ols','Shai-style affine OLS'),('ridge_alpha1','Fixed ridge sensitivity check'),('official_ce','Official NextLat linear CE heads')]:
        text.append(f"| {label} | {formatted(aggregate['dense_response_gpt']['sites']['final_norm']['64'][m]['bit_accuracy'])} | {formatted(aggregate['dense_response_nextlat']['sites']['final_norm']['64'][m]['bit_accuracy'])} |")
    text += ['### Every experimental condition at delay 64',
      '| Condition | Native recall (%) | Affine bit probe (%) | Official CE bit probe (%) | Training gate |',
      '|---|---:|---:|---:|---:|']
    for condition,a in aggregate.items():
        text.append(f"| {condition} | {formatted(a['native_recall']['64'])} | {formatted(a['sites']['final_norm']['64']['affine_ols']['bit_accuracy'])} | {formatted(a['sites']['final_norm']['64']['official_ce']['bit_accuracy'])} | {a['gates']}/3 |")
    text += ['### Methods and provenance',
      '- Shai et al.: affine least-squares decoding, adapted from belief-state regression to known binary cue marginals; not a replication of HMM belief-simplex geometry.',
      '- NextLat: reuse the unchanged released Probe.linear_probes heads (four bias-free binary readouts). Targets, L2 penalty and optimizer are adaptations; this is not the TinyStories future-token experiment.',
      '- Training-only standardization for affine/ridge; CE uses raw final-normalized states. All probes train on separate families at delays 4,8,16 (3,072 examples). No test tuning or checkpoint selection.',
      '- Shuffled-label controls permute both families and rows within families; 20 fixed fits per affine method/site/model. Test labels remain true. This is an adaptation of shuffle controls, not the original paper\'s full resampling protocol.',
      '- Embedding, both residual blocks, concatenated blocks and final norm are all reported. The final norm reproduces saved features bitwise across 264 run/split/delay checks.',
      '### Interpretation and limits',
      'The consistent result is stronger linear accessibility of relevant historical cues in response-supervised NextLat at long delay. In the dense condition, the second block carries markedly more decodable cue information than the first. At a trained delay, irrelevant nuisance-tag decoding is lower for NextLat, consistent with selective representation, but failed linear decoding does not prove erasure.',
      'Native recall at delay 64 is 60.3 +/- 12.8% for NTP and 79.9 +/- 4.5% for NextLat. NTP performs better at delays 24,32,48, so this is not a universal ranking. Probe accuracy is separate from unrestricted native generation.',
      'There are only three seeds. Test families were previously examined. All models were uniformly extended from 2,000 to 6,000 updates after validation learning failures, with early test results already visible; intermediate results remain preserved. Dense/full conditions do not all pass the >=95% learning gate at every training delay. Linear CE optimization uses a fixed 200-iteration cap and some fits do not attain the requested gradient tolerance. OLS and ridge independently support the main pattern.',
      'An ideal belief state does not discard predictive information. These experiments do not establish harmful compression, a hard state-capacity bottleneck, or a causal link between the auxiliary predictor and memory. Full causal attention remains available in normal Transformer inference. Attention-free latent rollout is separate evidence of retained information, not an equal-compute baseline.',
      '### Next steps',
      '1. Before submission: verify the figures/tables against the raw files, include both names, and submit an exploratory midterm report with the tested question and current limitations.',
      '2. After check-in: lock a fresh held-out set and endpoint, add training seeds, and test a response-supervised lambda sweep (0,.1,1,10) with stable in-distribution learning.',
      '3. Vary retention delay and cue capacity while holding prefix length and vocabulary fixed. Predictor-MLP width is an auxiliary-function capacity variable, not a hard Transformer state bottleneck.',
      '4. Add nonlinear probes and causal cue interventions to separate linear inaccessibility from actual information loss; retain all seeds and report learning gates.',
      '### References',
      '- Shai et al., Transformers Represent Belief State Geometry in their Residual Stream. https://arxiv.org/abs/2405.15943',
      '- Teoh et al., Next-Latent Prediction Transformers Learn Compact World Models, v4. https://arxiv.org/abs/2511.05963',
      '- Official probe code, pinned commit 3770be6: https://github.com/JaydenTeoh/NextLat/blob/3770be6009cea2b3c455a9ce7f2ca88b504bb955/models/model_probe.py',
      '### Reproduce',
      'From outputs/nextlat-published with the project Python environment:',
      '```sh\npython -m unittest memory.test_published_probes -v\npython -m memory.published_probes --run memory/runs/controls_v3_6000 --out memory/probes_midterm_reproduction\npython -m memory.probe_report --out memory/probes_midterm_reproduction\n```',
      'Full settings, source hashes, per-seed results, all layers and delays, fitted probe weights and optimizer diagnostics are retained alongside this report. The fitting script rejects a changed protocol when resuming.']
    rendered = ''
    previous = ''
    for line in text:
        rendered += ('\n' if line.startswith('|') and previous.startswith('|') else '\n\n') + line
        previous = line
    (out/'MIDTERM_RESULTS.md').write_text(rendered.lstrip()+'\n')
    print(out/'MIDTERM_RESULTS.md')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    generate(p.parse_args().out)
