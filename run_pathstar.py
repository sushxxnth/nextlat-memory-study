"""Thin single-device runner using unchanged official NextLat source code."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream'
PIN = '3770be6009cea2b3c455a9ce7f2ca88b504bb955'
sys.path.insert(0, str(UPSTREAM))
import torch
import lightning as L
from omegaconf import OmegaConf
from data.stargraph import Tokenizer, StarGraphDataset
from models.model_gpt import GPT, GPTConfig
from models.model_nextlat import NextLat, NextLatConfig

spec = importlib.util.spec_from_file_location('official_prepare', UPSTREAM / 'data/stargraph/prepare.py')
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


def verify_upstream():
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if actual != PIN or dirty:
        raise RuntimeError('Upstream must be the pinned, unchanged official checkout.')
    return actual


def encode(lines, c):
    tokenizer = Tokenizer(c['max_nodes'])
    dataset = StarGraphDataset(lines, tokenizer, c['path_length'], c['num_paths'])
    return torch.stack([dataset[i] for i in range(len(dataset))])


def make_data(c, directory):
    directory.mkdir(parents=True, exist_ok=True)
    cfg_path = directory / 'data_config.json'
    dc = {k: c[k] for k in ['num_paths','path_length','max_nodes','train_examples','validation_examples','test_examples']}
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != dc:
            raise ValueError('Data directory belongs to a different configuration.')
    else:
        prepare_module.generate_and_save_star_or_sink_graph_data(
            numOfSamples=c['train_examples'], numOfTestSamples=c['test_examples'],
            numOfPathsFromSource=c['num_paths'], lenOfEachPath=c['path_length'],
            maxNodes=c['max_nodes'], data_dir=str(directory), generate_test_data=True,
            shuffle_edge_lists=True, reverse=False, showLoadingBar=False)
        # Extra development set, independent of official sequential train/test seeds.
        lines=[]
        for i in range(c['validation_examples']):
            random.seed(10000000+i)
            edges,path,source,goal=prepare_module.star_or_sink_graph_maker(c['num_paths'],c['path_length'],c['max_nodes'],True)
            lines.append('|'.join(f'{a},{b}' for a,b in edges)+f'/{source},{goal}='+','.join(map(str,path)))
        (directory/'validation.txt').write_text('\n'.join(lines)+'\n')
        cfg_path.write_text(json.dumps(dc,indent=2))
    names={'train':f'graph_{c["num_paths"]}_{c["path_length"]}_sample_{c["train_examples"]}.txt',
           'test':f'graph_{c["num_paths"]}_{c["path_length"]}_test_{c["test_examples"]}.txt',
           'validation':'validation.txt'}
    sets={k:(directory/v).read_text().splitlines() for k,v in names.items()}
    if any(set(sets[a]) & set(sets[b]) for a,b in [('train','test'),('train','validation'),('validation','test')]):
        raise AssertionError('Duplicate graph strings across splits.')
    manifest={k:{'file':v,'sha256':hashlib.sha256((directory/v).read_bytes()).hexdigest(),'examples':len(sets[k])} for k,v in names.items()}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {k:encode(v,c) for k,v in sets.items()}, sets


def config_for(c, seq, nextlat=False):
    eq=c['max_nodes']+1
    position=int((seq[0]==eq).nonzero()[0])
    kw=dict(n_layer=c['layers'],n_head=c['heads'],n_embd=c['width'],dropout=0.,bias=False,
            vocab_size=c['max_nodes']+6,block_size=seq.shape[1],context_length=position,
            eos_token_id=c['max_nodes']+4,use_fused=False)
    if nextlat:
        return NextLatConfig(**kw,mtp_horizon=c['horizon'],lambda_mse=c['lambda_mse'],
                             lambda_kl=c['lambda_kl'],lambda_ce=0.,proj_factor=c['proj_factor'])
    return GPTConfig(**kw)


def build(c, seq, algorithm, fabric):
    # Both arms start with exactly the same transformer and output-head weights.
    torch.manual_seed(c['seed'])
    base=GPT(config_for(c,seq))
    if algorithm=='nextlat':
        torch.manual_seed(c['seed']+1)
        model=NextLat(config_for(c,seq,True))
        missing,unexpected=model.model.load_state_dict(base.model.state_dict(),strict=False)
        assert not unexpected and all(k.startswith('dynamics_model.') for k in missing)
    else:
        model=base
    with contextlib.redirect_stdout(io.StringIO()):
        model.configure_optimizers(c['weight_decay'],c['learning_rate'],(.9,.95),use_fused=False)
        model.setup_fabric(fabric)
    return model


def evaluate(model, data, c, fabric, split):
    dataset=StarGraphDataset([],Tokenizer(c['max_nodes']),c['path_length'],c['num_paths'])
    loader=torch.utils.data.DataLoader(data,batch_size=c.get('eval_batch_size',c['batch_size']),shuffle=False)
    loader=fabric.setup_dataloaders(loader)
    # Unchanged official evaluator: autoregressive sampling, full-path exact match.
    config=OmegaConf.create({'trainer':{'test_batches':0,'temperature':1.,'top_k':None}})
    torch.manual_seed(8000000+c['seed'])
    with contextlib.redirect_stdout(io.StringIO()):
        official=dataset.evaluate_stargraph(model,loader,config,fabric,show_progress_bar=False)
    model.eval()
    correct=first=total=0
    with torch.inference_mode():
        for b in loader:
            prefix=b[:,:-(c['path_length']+1)]
            target=b[:,-(c['path_length']+1):-1]
            generated=prefix.clone()
            for _ in range(c['path_length']):
                logits=model.model(generated)
                generated=torch.cat([generated,logits[:,-1].argmax(-1,keepdim=True)],dim=1)
            pred=generated[:,-c['path_length']:]
            correct+=int((pred==target).all(-1).sum())
            first+=int((pred[:,1]==target[:,1]).sum())
            total+=len(b)
    key=f'val_({c["num_paths"]}, {c["path_length"]})/test_accuracy'
    return {'split':split,'examples':total,'official_sampled_path_accuracy':official[key],
            'greedy_path_accuracy':correct/total,'greedy_first_branch_accuracy':first/total,
            'official_metrics':official}


def csv_write(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def backward_batch(model, batch, micro_batch_size):
    """Accumulate an effective batch without changing its loss normalization."""
    if len(batch) % micro_batch_size:
        raise ValueError('Microbatch size must divide the effective batch size.')
    chunks=len(batch)//micro_batch_size
    totals={}
    for i, micro in enumerate(batch.split(micro_batch_size)):
        values=model.compute_loss(micro,backpropagate=True,no_sync=i < chunks-1,loss_div=chunks)
        for key,value in values.items():
            if not torch.isfinite(value).all(): raise RuntimeError('Non-finite loss.')
            totals[key]=totals.get(key,0)+value
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/pilot.json'))
    p.add_argument('--algorithm',choices=['gpt','nextlat'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data-dir',default=str(ROOT/'datasets/pilot'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seed',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    commit=verify_upstream()
    c=json.loads(Path(args.config).read_text())
    if args.seed is not None: c['seed']=args.seed
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. Launch this job on an allocated GPU node.')
    precision=c.get('precision','32-true')
    if precision=='bf16-mixed' and (args.device!='cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('This profile requires a CUDA GPU supporting bf16.')
    micro_batch_size=c.get('micro_batch_size',c['batch_size'])
    if micro_batch_size<1 or c['batch_size']%micro_batch_size:
        raise ValueError('Microbatch size must divide the effective batch size.')
    if c['max_nodes'] < 1+c['num_paths']*(c['path_length']-1): raise ValueError('Too few node IDs.')
    if c['train_examples']+c['test_examples']>=10000000: raise ValueError('Training/test seeds would overlap validation.')
    torch.set_num_threads(c['threads'])
    out=Path(args.out)
    if not args.resume: out.mkdir(parents=True,exist_ok=False)
    tensors,_=make_data(c,Path(args.data_dir))
    fabric=L.Fabric(accelerator=args.device,devices=1,precision=precision)
    fabric.launch()
    model=build(c,tensors['train'],args.algorithm,fabric)
    params=sum(p.numel() for p in model.model.parameters())
    environment={'torch':str(torch.__version__),'lightning':L.__version__,'python':sys.version,'device':str(fabric.device),
                 'parameters':params,'upstream_commit':commit,'algorithm':args.algorithm,'sequence_length':tensors['train'].shape[1],
                 'context_length':model.config.context_length,'data_dir':str(Path(args.data_dir).resolve()),
                 'precision':precision,'micro_batch_size':micro_batch_size,
                 'gpu_name':torch.cuda.get_device_name() if args.device=='cuda' else None}
    gen=torch.Generator().manual_seed(c['seed']+9000)
    order=torch.randperm(len(tensors['train']),generator=gen);cursor=0;start=0;rows=[]
    if args.resume:
        state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if state['config']!=c or state['algorithm']!=args.algorithm: raise ValueError('Resume configuration mismatch.')
        model.model.module.load_state_dict(state['model']);model.optimizer.load_state_dict(state['optimizer'])
        gen.set_state(state['data_rng']);order=state['order'];cursor=state['cursor'];start=state['step'];rows=state['logs']
        model.training_steps=start
    (out/'config.json').write_text(json.dumps(c,indent=2))
    (out/'model_config.json').write_text(json.dumps(asdict(model.config),indent=2))
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    t0=time.monotonic()
    print(f'{args.algorithm}: {params:,} parameters; sequence length {tensors["train"].shape[1]}; {c["steps"]} steps',flush=True)
    for step in range(start+1,c['steps']+1):
        if cursor+c['batch_size']>len(order):
            order=torch.randperm(len(tensors['train']),generator=gen);cursor=0
        b=tensors['train'][order[cursor:cursor+c['batch_size']]].to(fabric.device);cursor+=c['batch_size']
        model.train()
        values=backward_batch(model,b,micro_batch_size)
        norm=model.optimizer_step(grad_clip=c['grad_clip'])
        rows.append({'step':step,**{k:float(v) for k,v in values.items()},'grad_norm':float(norm),'seconds_this_session':time.monotonic()-t0})
        if step==1 or step%c['eval_every']==0 or step==c['steps']:
            csv_write(out/'training.csv',rows)
            # Atomic checkpoint with optimizer and minibatch stream for resuming.
            state={'config':c,'algorithm':args.algorithm,'step':step,'model':model.model.module.state_dict(),
                   'optimizer':model.optimizer.state_dict(),'data_rng':gen.get_state(),'order':order,'cursor':cursor,'logs':rows}
            torch.save(state,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
            val=evaluate(model,tensors['validation'],c,fabric,'validation')
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps({'step':step,**val})+'\n')
            print(f'step={step} ntp={float(values["next_token_loss"]):.4f} sampled_path={val["official_sampled_path_accuracy"]:.3f} greedy_path={val["greedy_path_accuracy"]:.3f} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    test=evaluate(model,tensors['test'],c,fabric,'test')
    (out/'results.json').write_text(json.dumps({'status':'completed','scope':c.get('scope','scaled implementation pilot, not full paper reproduction'),**environment,'steps':c['steps'],'test':test},indent=2))
    print(json.dumps(test,indent=2),flush=True)


if __name__=='__main__':main()
