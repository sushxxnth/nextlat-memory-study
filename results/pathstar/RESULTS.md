# Published-task validation: Path-Star G(5,5)

**Aditi & Sushanth · November 12, 2026**

Both models completed 20,000 optimizer steps on an A100 and were evaluated on
20,000 held-out examples. The official sampled full-path accuracy was **18.625%
for NTP** and **98.630% for NextLat**, a gain of **80.005 percentage points**.
Greedy full-path accuracy was 18.965% and 99.500%, respectively.

| Method | Seed | Official sampled full-path accuracy | Greedy full-path accuracy |
|---|---:|---:|---:|
| NTP | 1234 | 18.625% | 18.965% |
| NTP + NextLat | 1234 | 98.630% | 99.500% |

## What this establishes

The official model, task and evaluation code produce the expected qualitative
separation under our runner. NTP stays near the 20% random-branch reference for a
five-arm graph, despite low teacher-forced training loss. NextLat learns to select
the correct path and generalizes to held-out graphs. This is a useful validation
before attributing behavior in our memory extension to the NextLat objective.
The random-branch reference assumes the remaining route is followed correctly;
it is not uniform random generation over the token vocabulary.

## Scope and evidence

One paired seed is preliminary evidence, not a multi-seed replication. We use
upstream commit `3770be6009cea2b3c455a9ce7f2ca88b504bb955` and our single-GPU
training runner, including gradient accumulation and a separate validation split.
Both result files say `completed`; each CSV has 20,000 training rows and each log
ends with `NEXTLAT_JOB_OK`. Slurm exit codes were not separately queried. Full configurations, environments, histories and logs are saved beside this report checkpoints remain on the cluster. No failures appear in the downloaded stderr files, only the Tensor Core precision advisory.

## Relation to our contribution

This experiment validates a published synthetic task. Our contribution is the
separate delayed-cue study of when compression preserves or loses relevant
history. Its three-seed CPU pilot is already complete, but loss-weight and
predictor-capacity sweeps have not run. Those sweeps, longer-delay evaluations,
and probes of individual hidden states are the next experiments. Strong NextLat
performance without a memory penalty remains a valid outcome.

## Suggested midterm wording

We first validated the implementation on the published Path-Star G(5,5) task
using the authors' model, generator and sampled-path evaluator. In one matched
seed, after 20,000 steps, NTP achieved 18.625% sampled full-path accuracy and
NextLat achieved 98.630% on 20,000 held-out examples. We then study task-relevant
memory in a separate delayed-cue environment using recall accuracy, frozen
single-state probes and attention-free latent rollouts. Our next experiments
vary the auxiliary-loss weight and predictor width and test longer delays.

## Paper cross-check (November 12, 2026)

Checked arXiv v4: Figure 6 on p. 9 (middle G(5,5) group), Table 5 on p. 29,
and Appendix F.3 on p. 30. The plot is visually near 20% GPT and near 100%
NextLat; no exact five-seed mean was transcribed from the plot. Our percentages
remain our single-seed measurements. Core run parameters match the released
G(5,5) YAMLs, including horizon 3 and predictor hidden width 384.

Reverification: all 8 implementation checks passed again; all source hashes
match provenance; training histories have 20,000 sequential finite rows per run;
stdout test dictionaries match saved JSON; downloaded artifacts match SHA256.
This was an artifact/code audit, not a new inference run from checkpoints.

Our runner differs through matched trunk initialization, separate validation,
microbatch accumulation, disabled compilation/fused optimizer execution, A100
hardware and evaluation of every held-out example (the released loader drops
the last incomplete batch). Marked source pages and audit JSON are in
`../../paper-verification/` relative to this report.
