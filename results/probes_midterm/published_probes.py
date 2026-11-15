"""Frozen-model probe robustness study; no backbone training or test tuning.

Shai et al.: affine least-squares decoding (adapted to four cue marginals).
Teoh et al.: directly reuse official bias-free linear readout modules, with
binary targets and a fixed convex optimizer instead of future-token training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from memory.experiment import L, build, fixed_data, verify_upstream
from memory.analysis import family_se
from memory.task import GROUP
from models.model_probe import Probe, ProbeConfig

DELAYS = [4, 8, 12, 16, 24, 32, 48, 64]
SITES = ['embedding', 'block1', 'block2', 'concat_blocks', 'final_norm']
NULL_REPEATS = 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(data, delay):
    bits = np.column_stack([data[f'delay{delay}/bit_{i}'] for i in range(4)])
    nuisance = np.eye(4)[data[f'delay{delay}/nuisance'].astype(int)]
    return np.column_stack([bits, nuisance]).astype(np.float64)


def shuffle_labels(y, seed):
    """Shuffle families AND rows within families, preserving joint labels.

    Family-only permutation is invalid: nuisance and queried-bit row order
    are deliberately identical across these counterfactual families.
    """
    rng = np.random.default_rng(seed)
    grouped = y.reshape(-1, GROUP, y.shape[1])
    order = rng.permutation(len(grouped))
    rows = np.stack([rng.permutation(GROUP) for _ in order])
    return grouped[order[:, None], rows].reshape(y.shape)


def score(pred, truth):
    correct = (pred[:, :4] >= .5) == truth[:, :4]
    return dict(bit_accuracy=float(correct.mean()),
                bit_accuracy_family_se=family_se(correct.mean(1)),
                bit_mse=float(np.mean((pred[:, :4]-truth[:, :4])**2)),
                bit_r2=float(1-np.sum((pred[:, :4]-truth[:, :4])**2)
                             / np.sum((truth[:, :4]-truth[:, :4].mean(0))**2)),
                nuisance_accuracy=float((pred[:, 4:].argmax(1) == truth[:, 4:].argmax(1)).mean()))


def affine_fit(x, y):
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    ys = np.column_stack([y] + [shuffle_labels(y, 91000+r) for r in range(NULL_REPEATS)])
    # Separable output objectives: one solve includes real and shuffled labels.
    return scaler, {name: estimator.fit(xs, ys) for name, estimator in
                    [('affine_ols', LinearRegression()), ('ridge_alpha1', Ridge(alpha=1.))]}


def affine_evaluate(scaler, fits, x, y):
    result = {}
    xs = scaler.transform(x)
    for name, fit in fits.items():
        prediction = fit.predict(xs).reshape(len(x), NULL_REPEATS+1, 8)
        result[name] = score(prediction[:, 0], y)
        nulls = [score(prediction[:, r+1], y)['bit_accuracy'] for r in range(NULL_REPEATS)]
        result[name]['shuffled_training_label_accuracy_mean'] = float(np.mean(nulls))
        result[name]['shuffled_training_label_accuracy_sd'] = float(np.std(nulls, ddof=1))
        result[name]['shuffled_training_label_accuracies'] = nulls
    return result


def ce_fit(x, y):
    torch.manual_seed(73021)
    # Use the upstream heads themselves. The upstream decoder is unused.
    heads = Probe(ProbeConfig(n_embd=x.shape[1], vocab_size=2, probe_depth=4)).linear_probes.double()
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y[:, :4], dtype=torch.long)
    optimizer = torch.optim.LBFGS(heads.parameters(), max_iter=200, max_eval=250,
                                  tolerance_grad=1e-8, tolerance_change=1e-10,
                                  line_search_fn='strong_wolfe')
    calls = 0

    def objective():
        logits = torch.stack([head(xt) for head in heads], dim=1)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), yt.reshape(-1))
        return loss + sum(p.square().sum() for p in heads.parameters())/(2*len(xt)*4)

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure()
    audit = dict(closure_calls=calls, objective=float(loss.detach()),
                 gradient_max=float(max(p.grad.abs().max() for p in heads.parameters())))
    assert np.isfinite(audit['objective'])
    return heads, audit


def ce_evaluate(heads, x, y):
    with torch.inference_mode():
        logits = torch.stack([head(torch.as_tensor(x, dtype=torch.float64)) for head in heads], dim=1)
        predicted = logits.argmax(-1).numpy()
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), torch.as_tensor(y[:, :4].reshape(-1), dtype=torch.long))
    correct = predicted == y[:, :4]
    return dict(bit_accuracy=float(correct.mean()), bit_accuracy_family_se=family_se(correct.mean(1)),
                binary_cross_entropy=float(ce))


@torch.inference_mode()
def extract(model, c, split, delays, cached):
    core = model.model.module
    current = {}
    hooks = [core.token_embedding.register_forward_hook(
        lambda module, args, output: current.__setitem__('embedding', output[:, -2].float().cpu().numpy().copy()))]
    for i, block in enumerate(core.transformer.blocks):
        hooks.append(block.register_forward_hook(
            lambda module, args, output, index=i: current.__setitem__(f'block{index+1}', output[:, -2].float().cpu().numpy().copy())))
    features = {}
    try:
        for delay in delays:
            seq, _ = fixed_data(c, split, delay, c['eval_examples'] if split == 'test' else c['probe_examples_per_delay'])
            parts = {site: [] for site in SITES}
            for chunk in seq.split(c['batch_size']):
                _, hidden = model.model(chunk[:, :-2], return_hidden_states=True)
                current['final_norm'] = hidden[:, -2].float().cpu().numpy()
                current['concat_blocks'] = np.concatenate([current['block1'], current['block2']], axis=1)
                for site in SITES:
                    parts[site].append(current[site])
            features[delay] = {site: np.concatenate(value) for site, value in parts.items()}
            np.testing.assert_array_equal(features[delay]['final_norm'], cached[f'delay{delay}/before_query'])
            # Query index is after the inspected token. Answer/EOS never input.
    finally:
        for hook in hooks:
            hook.remove()
    return features


def run(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source_plan = json.loads((root/'plan.json').read_text())
    protocol = dict(source_run=str(root.resolve()), source_plan_sha256=sha(root/'plan.json'),
        analysis_source_sha256=sha(__file__), seeds=source_plan['seeds'], delays=DELAYS,
        sites=SITES, shuffled_label_repeats=NULL_REPEATS,
        primary='Before-query final-normalized state, D=64, response-supervised models; affine OLS and official CE heads.',
        probe_training_delays=[4, 8, 16], test_reuse='Exploratory robustness on previously viewed test families; not fresh confirmation.',
        targets='Four cue-bit marginals, plus 4-way nuisance for affine probes. No future query side information.',
        affine='Training-only StandardScaler, affine OLS; fixed Ridge alpha=1 sensitivity check; threshold=.5.',
        ce='Official Probe.linear_probes, four bias-free binary heads, raw final-normalized features; LBFGS max_iter=200, fixed L2=sum(W^2)/(2*N*4).',
        shuffle='20 fixed permutations: whole families plus independent within-family rows; shuffled training labels, true test labels.',
        interpretation='Linear accessibility, not proof of information erasure. No test layer/probe selection; all layers and cells reported.')
    protocol_file = out/'protocol.json'
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text()) == protocol, 'Protocol/source changed; use a new output directory.'
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2)+'\n')
        (out/'published_probes.py').write_text(Path(__file__).read_text())
    verify_upstream()
    torch.set_num_threads(2)
    all_records = []
    with threadpool_limits(limits=2):
        for entry in source_plan['entries']:
            name = entry['relative_run'].replace('/', '__')
            target = out/name
            if (target/'results.json').exists():
                all_records.append(json.loads((target/'results.json').read_text()))
                continue
            folder = root/entry['relative_run']; c = entry['config']
            checkpoint = torch.load(folder/'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['step'] == c['steps'] == 6000 and checkpoint['config'] == c
            fabric = L.Fabric(accelerator='cpu', devices=1, precision='32-true'); fabric.launch()
            model = build(c, entry['algorithm'], fabric)
            model.model.module.load_state_dict(checkpoint['model']); model.eval()
            for p in model.model.parameters():
                p.requires_grad_(False)
            with np.load(folder/'probe_train_features.npz') as train_cache, np.load(folder/'test_features.npz') as test_cache:
                train = extract(model, c, 'probe_train', [4, 8, 16], train_cache)
                test = extract(model, c, 'test', DELAYS, test_cache)
                y = np.concatenate([labels(train_cache, d) for d in [4, 8, 16]])
                record = dict(condition=entry['condition'], algorithm=entry['algorithm'], seed=c['seed'],
                              checkpoint_sha256=sha(folder/'checkpoint.pt'),
                              feature_cache_sha256=sha(folder/'test_features.npz'),
                              cached_final_states_bitwise_equal=True, sites={})
                fitted = {}
                for site in SITES:
                    x = np.concatenate([train[d][site] for d in [4, 8, 16]])
                    scaler, fits = affine_fit(x, y)
                    fitted[site] = (scaler, fits)
                    record['sites'][site] = {str(d): affine_evaluate(scaler, fits, test[d][site], labels(test_cache, d)) for d in DELAYS}
                heads, audit = ce_fit(np.concatenate([train[d]['final_norm'] for d in [4, 8, 16]]), y)
                record['ce_fit_audit'] = audit
                for d in DELAYS:
                    record['sites']['final_norm'][str(d)]['official_ce'] = ce_evaluate(heads, test[d]['final_norm'], labels(test_cache, d))
                original = json.loads((folder/'results.json').read_text())
                record['native_recall'] = {str(d): original['test'][str(d)]['answer_accuracy'] for d in DELAYS}
                record['existing_logistic_probe'] = {str(d): original['test'][str(d)]['probes']['before_query/stored_bits_mean']['accuracy'] for d in DELAYS}
                record['full_training_delay_gate'] = all(original['test'][str(d)]['answer_accuracy'] >= .95 for d in c['train_delays'])
                target.mkdir(parents=True, exist_ok=True)
                joblib.dump(fitted, target/'affine_probes.joblib', compress=3)
                torch.save(heads.state_dict(), target/'official_ce_heads.pt')
                (target/'results.json').write_text(json.dumps(record, indent=2)+'\n')
                all_records.append(record)
                print(f"Completed {entry['condition']} seed{c['seed']} OLS64={record['sites']['final_norm']['64']['affine_ols']['bit_accuracy']:.4f} CE64={record['sites']['final_norm']['64']['official_ce']['bit_accuracy']:.4f}", flush=True)
    (out/'all_results.json').write_text(json.dumps(all_records, indent=2)+'\n')
    return all_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.out)
