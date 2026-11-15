"""Verify included evidence hashes and recompute reported probe aggregates."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'results/ARTIFACT_SHA256.json').read_text())
    for name, expected in manifest.items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SystemExit(f'Evidence hash mismatch: {name}')
    folder=ROOT/'results/probes_midterm'
    raw=json.loads((folder/'all_results.json').read_text())
    aggregate=json.loads((folder/'aggregate.json').read_text())
    if len(raw)!=24:
        raise SystemExit('Expected all 24 models.')
    checks=0
    for condition, group in aggregate.items():
        rows=sorted([r for r in raw if r['condition']==condition],key=lambda r:r['seed'])
        if [r['seed'] for r in rows]!=[1237,1238,1239]:
            raise SystemExit(f'Seed mismatch: {condition}')
        for site, delays in group['sites'].items():
            for delay, methods in delays.items():
                for method, metrics in methods.items():
                    values=[r['sites'][site][delay][method]['bit_accuracy'] for r in rows]
                    target=metrics['bit_accuracy']
                    if not (math.isclose(statistics.mean(values),target['mean'],abs_tol=1e-12)
                            and math.isclose(statistics.stdev(values),target['sd'],abs_tol=1e-12)):
                        raise SystemExit(f'Aggregate mismatch: {condition}/{site}/{delay}/{method}')
                    checks+=1
    baseline=json.loads((ROOT/'results/pathstar/comparison.json').read_text())
    for model, expected in [('gpt',.18625),('nextlat',.9863)]:
        r=baseline[model]
        if r['steps']!=20000 or r['test_examples']!=20000 or not math.isclose(r['sampled_full_path_accuracy'],expected,abs_tol=1e-7):
            raise SystemExit(f'Unexpected baseline record: {model}')
    print(f'PASS: {len(manifest)} evidence hashes; {checks} probe aggregates across 24 models; both baseline records.')
    print('This verifies included records, not checkpoint inference or fresh generalization.')


if __name__=='__main__':
    main()
