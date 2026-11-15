"""Delayed-memory experiment using unchanged official GPT/NextLat code.

Run from nextlat-published: python -m memory.experiment --out memory/runs/pilot
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import time
from dataclasses import asdict
from pathlib import Path
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from run_pathstar import (ROOT, PIN, L, GPT, GPTConfig, NextLat, NextLatConfig,
                          verify_upstream, backward_batch, csv_write)
from memory import task
from memory.analysis import family_se, fit_probes, geometry, probe_metrics


def validate(c):
    if c.get('loss_scope', 'full') not in ['full', 'response']:
        raise ValueError('loss_scope must be full or response.')
    if not isinstance(c.get('num_cues', 1), int) or c.get('num_cues', 1) < 1:
        raise ValueError('num_cues must be a positive integer.')
    if c['width'] % c['heads'] or (c['width']//c['heads']) % 2:
        raise ValueError('RoPE needs even head width.')
    for key in ['batch_size', 'eval_examples', 'probe_examples_per_delay']:
        if c[key] < task.GROUP or c[key] % task.GROUP:
            raise ValueError(f'{key} must be divisible by eight.')
    if min(c['train_delays']+c['eval_delays']) < 4:
        raise ValueError('Minimum delay is four.')
    for key in ['probe_delays', 'validation_delays']:
        if key in c and (not c[key] or not set(c[key]) <= set(c['train_delays'])):
            raise ValueError(f'{key} must be a nonempty subset of training delays.')
    hidden = 128*round(c['width']*2*c['proj_factor']/128)
    if hidden < 128:
        raise ValueError('Official rounding would create an empty dynamics MLP.')
    if c['horizon'] >= min(c['train_delays'])+4:
        raise ValueError('Prediction horizon must leave supervised tokens.')
    if c['lambda_aux'] < 0 or c['steps'] < 1 or c['eval_every'] < 1:
        raise ValueError('Invalid loss weight or step count.')


def model_config(c, algorithm):
    num_cues = c.get('num_cues', 1)
    kw = dict(block_size=max(c['train_delays']+c['eval_delays'])+num_cues+5,
              vocab_size=task.vocab_size(num_cues), n_layer=c['layers'], n_head=c['heads'],
              n_embd=c['width'], dropout=0., bias=False, context_length=0,
              eos_token_id=task.EOS, use_fused=False)
    if algorithm == 'nextlat':
        return NextLatConfig(**kw, mtp_horizon=c['horizon'],
                             lambda_mse=c['lambda_aux'], lambda_kl=c['lambda_aux'],
                             lambda_ce=0., proj_factor=c['proj_factor'])
    return GPTConfig(**kw)


def build(c, algorithm, fabric):
    torch.manual_seed(c['seed'])
    base = GPT(model_config(c, 'gpt'))
    model = base
    if algorithm == 'nextlat':
        torch.manual_seed(c['seed']+1)
        model = NextLat(model_config(c, algorithm))
        missing, unexpected = model.model.load_state_dict(base.model.state_dict(), strict=False)
        assert not unexpected and all(k.startswith('dynamics_model.') for k in missing)
    with contextlib.redirect_stdout(io.StringIO()):
        model.configure_optimizers(c['weight_decay'], c['learning_rate'], (.9, .95), use_fused=False)
        model.setup_fabric(fabric)
    return model


def fixed_data(c, split, delay, n):
    seeds = {'probe_train': 2000000, 'validation': 3000000, 'test': 4000000}
    generator = torch.Generator().manual_seed(seeds[split]+c['seed']*1009+delay)
    return task.batch(n, delay, generator, split, c.get('num_cues', 1))


@contextlib.contextmanager
def loss_context(model, seq, c):
    """Use the official prompt mask per uniform-length batch, then restore it.

    RESPONSE supervises ANSWER and EOS with NTP/KL; latent Smooth L1 still
    covers context transitions under the authors' unchanged implementation.
    """
    previous = model.config.context_length
    boundary = seq.shape[1]-3 if c.get('loss_scope', 'full') == 'response' else 0
    model.config.context_length = boundary
    try:
        yield boundary
    finally:
        model.config.context_length = previous


@torch.inference_mode()
def observe(model, seq, algorithm, device, num_cues=1):
    # Neither the answer nor EOS is passed to the transformer at evaluation.
    prefix = seq[:, :-2].to(device)
    output, h = model.model(prefix, return_hidden_states=True)
    core = model.model.module
    logits = core.lm_head(h[:, -1])
    target = seq[:, -2].to(device)
    ok = (logits.argmax(-1) == target).cpu().numpy()
    result = dict(answer_accuracy=float(ok.mean()), answer_accuracy_family_se=family_se(ok),
                  non_answer_token_rate=float(((logits.argmax(-1) != task.ANSWER_RED)
                                                & (logits.argmax(-1) != task.ANSWER_BLUE)).float().mean()),
                  answer_ce=float(F.cross_entropy(logits.float(), target)),
                  constrained_answer_accuracy=float((logits[:, task.ANSWER_RED:task.ANSWER_BLUE+1].argmax(-1)
                                                     == target-task.ANSWER_RED).float().mean()))
    result['_correct'] = {'answer': ok}
    result['_n'] = len(seq)
    anchor = task.anchor_index(num_cues)
    features = {'anchor': h[:, anchor].float().cpu().numpy(),
                'before_query': h[:, -2].float().cpu().numpy(),
                'at_query': h[:, -1].float().cpu().numpy()}
    if algorithm == 'nextlat':
        # Start after the first distractor. The only future inputs are observed
        # distractor/query embeddings; no answer, cached attention, or new trunk call.
        state = h[:, anchor]
        for t in range(anchor+1, prefix.shape[1]):
            state = core.dynamics_model(state, core.token_embedding(prefix[:, t]))
        one = core.dynamics_model(h[:, -2], core.token_embedding(prefix[:, -1]))
        result['latent_rollout_steps'] = prefix.shape[1]-anchor-1
        for label, latent in [('rollout', state), ('one_step', one)]:
            pred = core.lm_head(latent).argmax(-1)
            correct = (pred == target).cpu().numpy()
            result[f'latent_{label}_accuracy'] = float(correct.mean())
            result[f'latent_{label}_accuracy_family_se'] = family_se(correct)
            result['_correct'][f'latent_{label}'] = correct
            result[f'latent_{label}_teacher_state_mse'] = float(F.mse_loss(latent.float(), h[:, -1].float()))
        features['latent_query'] = state.float().cpu().numpy()
    return result, features


def measure(model, c, algorithm, fabric, split, n, delays):
    model.eval()
    records = {}
    for delay in delays:
        seq, keys = fixed_data(c, split, delay, n)
        scores, features = [], {}
        for chunk in seq.split(c['batch_size']):
            with fabric.autocast():
                score, hidden = observe(model, chunk, algorithm, fabric.device, c.get('num_cues', 1))
            scores.append(score)
            for site, value in hidden.items(): features.setdefault(site, []).append(value)
        features = {k: np.concatenate(v) for k, v in features.items()}
        metrics = {k: float(sum(score[k]*score['_n'] for score in scores)/n)
                   for k in scores[0] if not k.startswith('_') and not k.endswith('_family_se')}
        for label in scores[0]['_correct']:
            correct = np.concatenate([score['_correct'][label] for score in scores])
            metrics[f'{label}_accuracy_family_se'] = family_se(correct)
        records[delay] = {'metrics': metrics, 'features': features,
                          'labels': task.labels(seq, c.get('num_cues', 1)), 'families': keys,
                          'geometry': {site: geometry(value) for site, value in features.items()}}
    return records


def save_features(path, parts):
    arrays = {}
    for delay, part in parts.items():
        for site, x in part['features'].items(): arrays[f'delay{delay}/{site}'] = x
        for label, y in part['labels'].items(): arrays[f'delay{delay}/{label}'] = y
    np.savez_compressed(path, **arrays)


def run(c, algorithm, out, device, resume):
    if (out/'results.json').exists():
        if json.loads((out/'config.json').read_text()) != c:
            raise ValueError('Existing completed run has a different configuration.')
        saved = json.loads((out/'results.json').read_text())
        if saved.get('status') != 'completed' or saved.get('algorithm') != algorithm:
            raise ValueError('Existing results are incomplete or belong to another algorithm.')
        print(f'Skipping completed {out}', flush=True)
        return
    if not resume: out.mkdir(parents=True, exist_ok=False)
    elif not (out/'checkpoint.pt').exists(): raise ValueError('No checkpoint to resume.')
    fabric = L.Fabric(accelerator=device, devices=1, precision=c['precision']); fabric.launch()
    model = build(c, algorithm, fabric)
    core = model.model.module
    generator = torch.Generator().manual_seed(c['seed']+9000)
    rows, start, data_digest = [], 0, ''
    if resume:
        ckpt = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False)
        if ckpt['config'] != c or ckpt['algorithm'] != algorithm: raise ValueError('Resume mismatch.')
        core.load_state_dict(ckpt['model']); model.optimizer.load_state_dict(ckpt['optimizer'])
        generator.set_state(ckpt['data_rng']); rows=ckpt['logs']; start=ckpt['step']
        model.training_steps=start
        data_digest = ckpt.get('training_data_digest', '')
    (out/'config.json').write_text(json.dumps(c, indent=2))
    (out/'model_config.json').write_text(json.dumps(asdict(model.config), indent=2))
    t0=time.monotonic()
    print(f'{algorithm}: {sum(p.numel() for p in core.parameters()):,} parameters, {c["steps"]} steps', flush=True)
    for step in range(start+1, c['steps']+1):
        delay = c['train_delays'][int(torch.randint(len(c['train_delays']), (), generator=generator))]
        seq, _ = task.batch(c['batch_size'], delay, generator, 'train', c.get('num_cues', 1))
        data_digest = hashlib.sha256(data_digest.encode()+seq.numpy().tobytes()).hexdigest()
        model.train()
        with loss_context(model, seq, c) as boundary:
            values = backward_batch(model, seq.to(fabric.device), c['batch_size'])
        norm = model.optimizer_step(grad_clip=c['grad_clip'])
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient norm.')
        rows.append(dict(step=step, delay=delay, **{k: float(v) for k, v in values.items()},
                         grad_norm=float(norm), seconds_this_session=time.monotonic()-t0))
        if 'loss_scope' in c:
            rows[-1].update(token_context_length=boundary,
                            supervised_ntp_tokens_per_example=2 if boundary else seq.shape[1]-1)
        if step == 1 or step % c['eval_every'] == 0 or step == c['steps']:
            csv_write(out/'training.csv', rows)
            state=dict(config=c, algorithm=algorithm, step=step, model=core.state_dict(),
                       optimizer=model.optimizer.state_dict(), data_rng=generator.get_state(), logs=rows,
                       training_data_digest=data_digest)
            torch.save(state, out/'checkpoint.tmp'); (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
            val=measure(model,c,algorithm,fabric,'validation',c['eval_examples'],c.get('validation_delays', c['train_delays']))
            summary={d: p['metrics'] for d,p in val.items()}
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step, delays=summary))+'\n')
            print(f'step {step}: ntp={float(values["next_token_loss"]):.4f}, val recall={np.mean([p["answer_accuracy"] for p in summary.values()]):.3f}', flush=True)
    probe_train=measure(model,c,algorithm,fabric,'probe_train',c['probe_examples_per_delay'],c.get('probe_delays', c['train_delays']))
    probes=fit_probes(probe_train)
    test=measure(model,c,algorithm,fabric,'test',c['eval_examples'],c['eval_delays'])
    save_features(out/'probe_train_features.npz',probe_train); save_features(out/'test_features.npz',test)
    joblib.dump(probes,out/'probes.joblib')
    manifest={split:{str(d):p['families'] for d,p in parts.items()} for split,parts in [('probe_train',probe_train),('test',test)]}
    (out/'families.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='completed', scope=c.get('scope', 'single-seed controlled memory pilot, not a confirmed research finding'),
                algorithm=algorithm, seed=c['seed'], steps=c['steps'], source_commit=PIN,
                parameters=sum(p.numel() for p in core.parameters()),
                num_cues=c.get('num_cues', 1), training_data_digest=data_digest,
                predictor_hidden_width=(core.dynamics_model.mlp[0].out_features if algorithm=='nextlat' else None),
                environment=dict(torch=str(torch.__version__), lightning=L.__version__, device=str(fabric.device)),
                final_batch_ntp=rows[-1]['next_token_loss'], seconds_this_session=time.monotonic()-t0,
                test={str(d):dict(**p['metrics'], geometry=p['geometry'], probes=probe_metrics(probes,p),
                                 in_training_delays=d in c['train_delays']) for d,p in test.items()})
    (out/'results.tmp').write_text(json.dumps(result,indent=2)); (out/'results.tmp').replace(out/'results.json')
    print(f'Completed {algorithm}: {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'memory/configs/pilot.json'))
    p.add_argument('--out',required=True)
    p.add_argument('--algorithm',choices=['both','gpt','nextlat'],default='both')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--lambda-aux',type=float)
    p.add_argument('--proj-factor',type=float)
    p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); verify_upstream()
    c=json.loads(Path(args.config).read_text())
    for key in ['lambda_aux','proj_factor','steps','seed']:
        if getattr(args,key) is not None: c[key]=getattr(args,key)
    validate(c)
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('No allocated CUDA GPU.')
    if c['precision']=='bf16-mixed' and (args.device!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('This precision requires a bf16-compatible GPU.')
    torch.set_num_threads(c['threads'])
    algorithms=['gpt','nextlat'] if args.algorithm=='both' else [args.algorithm]
    out=Path(args.out)
    with threadpool_limits(limits=c['threads']):
        for algorithm in algorithms: run(c,algorithm,out/algorithm,args.device,args.resume)


if __name__=='__main__': main()
