"""Frozen linear probes and matched-history geometry; no test-set fitting."""
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from memory.task import GROUP, TAGS


def family_se(correct):
    values = np.asarray(correct, dtype=float).reshape(-1, GROUP).mean(1)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.


def geometry(h):
    x = np.asarray(h, dtype=np.float64)
    norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    groups = norm.reshape(-1, 2, TAGS, x.shape[1])
    within = [np.sum(groups[:, cue, a]*groups[:, cue, b], axis=-1)
              for cue in range(2) for a, b in combinations(range(TAGS), 2)]
    between = np.sum(groups[:, 0]*groups[:, 1], axis=-1)
    spectrum = np.linalg.svd(x-x.mean(0), compute_uv=False)**2
    p = spectrum / max(float(spectrum.sum()), 1e-30)
    rank = float(np.exp(-np.sum(p[p > 0]*np.log(p[p > 0])))) if spectrum.sum() > 1e-20 else 0.
    return dict(same_cue_different_tag_cosine=float(np.mean(within)),
                different_cue_same_tag_cosine=float(between.mean()),
                covariance_entropy_effective_rank=rank)


def fit_probes(training):
    fitted = {}
    for site in next(iter(training.values()))['features']:
        x = np.concatenate([part['features'][site] for part in training.values()])
        targets = [key for key in next(iter(training.values()))['labels'] if key != 'query_index']
        for target in targets:
            # The future query is unknown before QUERY. A pooled probe of its
            # selected answer would confound retrieval with an unknown index.
            if target == 'cue' and 'query_index' in next(iter(training.values()))['labels'] and site in ['anchor', 'before_query']:
                continue
            y = np.concatenate([part['labels'][target] for part in training.values()])
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1500, random_state=0))
            probe.fit(x, y)
            fitted[site, target] = probe
    return fitted


def probe_metrics(fitted, part):
    results = {}
    for (site, target), probe in fitted.items():
        correct = probe.predict(part['features'][site]) == part['labels'][target]
        results[f'{site}/{target}'] = dict(accuracy=float(correct.mean()), family_se=family_se(correct))
    bits = sorted([key for key in part['labels'] if key.startswith('bit_')], key=lambda k: int(k[4:]))
    if bits:
        for site in part['features']:
            predicted = np.stack([fitted[site, bit].predict(part['features'][site]) for bit in bits], axis=1)
            truth = np.stack([part['labels'][bit] for bit in bits], axis=1)
            per_example = (predicted == truth).mean(axis=1)
            results[f'{site}/stored_bits_mean'] = dict(accuracy=float(per_example.mean()), family_se=family_se(per_example))
            selected = predicted[np.arange(len(predicted)), part['labels']['query_index']]
            correct = selected == part['labels']['cue']
            results[f'{site}/selected_bit_given_query'] = dict(accuracy=float(correct.mean()), family_se=family_se(correct))
    return results
