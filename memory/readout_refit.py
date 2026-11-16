"""Refit only the existing full-vocabulary output head on held-out probe data.

This diagnoses accessible information at the query state, where full Transformer
attention has already occurred. It does not establish memory in a pre-query
single state, or prove that a low-scoring representation erased information.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from memory.task import ANSWER_RED, ANSWER_BLUE


def head_metrics(features, targets, weight):
    with torch.inference_mode():
        logits = torch.as_tensor(features, dtype=torch.float64) @ weight.T
        predicted = logits.argmax(-1).numpy()
        valid = (predicted == ANSWER_RED) | (predicted == ANSWER_BLUE)
        restricted = logits[:, ANSWER_RED:ANSWER_BLUE+1].argmax(-1).numpy()+ANSWER_RED
    correct = predicted == targets
    return dict(accuracy=float(correct.mean()), non_answer_rate=float((~valid).mean()),
                restricted_accuracy=float((restricted == targets).mean()),
                correct_given_answer_token=float(correct[valid].mean()) if valid.any() else None)


def fit_full_head(features, targets, original, max_iter=300):
    x = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)
    initial = torch.as_tensor(original, dtype=torch.float64).clone()
    weight = torch.nn.Parameter(initial.clone())
    # Identical head architecture; initialize from native weights. Only this
    # matrix is optimized. L2 around native weights is fixed before evaluation.
    optimizer = torch.optim.LBFGS([weight], max_iter=max_iter, max_eval=max_iter*2,
        line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-10)
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x@weight.T, y)
        loss = loss+(weight-initial).square().sum()/(2*len(x))
        loss.backward()
        return loss

    optimizer.step(closure)
    value = closure()
    audit = dict(objective=float(value.detach()), gradient_max=float(weight.grad.abs().max()),
                 closure_calls=calls, max_iterations=max_iter,
                 training=head_metrics(features, targets, weight.detach()))
    if not np.isfinite(audit['objective']):
        raise RuntimeError('Nonfinite head-refit objective.')
    return weight.detach(), audit


def refit_run(folder):
    folder = Path(folder)
    target = folder/'readout_refit.json'
    if target.exists():
        return json.loads(target.read_text())
    checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
    c = checkpoint['config']
    original = checkpoint['model']['lm_head.weight'].double()
    manifest = json.loads((folder/'families.json').read_text())
    train_keys = {k for keys in manifest['probe_train'].values() for k in keys}
    test_keys = {k for keys in manifest['test'].values() for k in keys}
    if train_keys & test_keys:
        raise RuntimeError('Probe-fitting and test families overlap.')
    with np.load(folder/'probe_train_features.npz') as train, np.load(folder/'test_features.npz') as test:
        delays = c.get('probe_delays', c['train_delays'])
        x = np.concatenate([train[f'delay{d}/at_query'] for d in delays])
        y = np.concatenate([train[f'delay{d}/cue']+ANSWER_RED for d in delays])
        fitted, audit = fit_full_head(x, y, original)
        result = dict(scope='Frozen Transformer; refit native bias-free, full-vocabulary head on separate probe families. No forced-choice evaluation.',
            checkpoint_sha256=hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest(),
            fitting_delays=delays, fitting_examples=len(x), output_classes=original.shape[0],
            regularization='sum((W-W_native)^2)/(2*N)', optimizer=audit, test={})
        for d in c['eval_delays']:
            h = test[f'delay{d}/at_query']; labels=test[f'delay{d}/cue']+ANSWER_RED
            result['test'][str(d)] = dict(native_cpu_redecode=head_metrics(h, labels, original),
                                         refitted=head_metrics(h, labels, fitted))
    torch.save(fitted, folder/'refitted_output_head.pt')
    target.write_text(json.dumps(result, indent=2)+'\n')
    return result
