# NextLat Memory Study

**Aditi & Sushanth** | The Science of Deep Learning, Fall 2026

We compare standard next-token prediction (NTP) with NextLat to study how well
Transformers retain useful information from earlier in a sequence.

## What we test

- **Published baseline:** Path-Star G(5,5), using the official NextLat code.
- **Our experiment:** four red/blue cues, random distractors, and a later query
  asking for one cue. Probes test which cues can be recovered from a single
  hidden state before the query arrives.

Our memory study has 24 models across two objectives, two training-delay
schedules, two supervision settings, and three seeds. Each model receives
192,000 sequence presentations: 6,000 updates with batch size 32.

## Results

On Path-Star, sampled full-path accuracy was **18.625% for NTP** and
**98.630% for NextLat** in one matched seed.

For the four-cue task below, models train on delays 4–16 with response
supervision and are evaluated at delay 64:

| Metric | NTP | NextLat |
|---|---:|---:|
| Affine cue probe | 56.3% ± 4.0% | 83.4% ± 11.3% |
| Ridge cue probe | 55.5% ± 6.0% | 86.4% ± 11.0% |
| Linear CE cue probe | 57.0% ± 6.0% | 82.7% ± 7.1% |
| Native answer accuracy | 60.3% ± 12.8% | 79.9% ± 4.5% |

Values are mean ± sample SD across three seeds. Cue-probe chance is 50%.
These preliminary results support useful memory preservation in this setting.
They do not establish harmful compression or a universal NextLat advantage:
NTP has higher native answer accuracy at delays 24, 32 and 48.

## Setup

Use Python 3.12:

```sh
git clone --recurse-submodules https://github.com/sushxxnth/nextlat-memory-study.git
cd nextlat-memory-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The official NextLat dependency is pinned in `upstream/`.

## Run

Short four-cue pipeline check:

```sh
python -m memory.experiment --config configs/memory_four_cue_smoke.json --out runs/four_cue_smoke
```

Verify the recorded results:

```sh
python scripts/verify_recorded_results.py
```

## More details

- [Methods, tests, and full reproduction commands](docs/REPRODUCING.md)
- [Detailed results and limitations](docs/MIDTERM_RESULTS.md)
- [Recorded experiment files](results/README.md)
- [Source attribution](THIRD_PARTY_NOTICES.md)

The analysis is exploratory: test data had previously been examined, training
was extended uniformly from 2,000 to 6,000 updates, and some conditions did not
fully learn the training task. Checkpoints and generated datasets are kept out
of Git. See the linked documents for the complete experimental record.
