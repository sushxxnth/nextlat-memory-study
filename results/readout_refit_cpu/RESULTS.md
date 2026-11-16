# Frozen-head intervention on existing CPU models

Exploratory follow-up on the original 24 final 6,000-update CPU checkpoints. These are not new GPU results. All checkpoint hashes remain unchanged and native full-vocabulary head scores exactly reproduce the recorded delay-64 accuracies. Values are mean +/- sample SD across three training seeds.

| Condition | Native recall | Frozen-head refit recall | Paired change (pp) | Native non-answer rate | Refit non-answer rate | Learning gate |
|---|---:|---:|---:|---:|---:|---:|
| dense_full_gpt | 18.2 +/- 19.7 | 26.6 +/- 31.7 | 8.4 +/- 12.4 | 66.0 +/- 35.9 | 54.8 +/- 50.7 | 2/3 |
| dense_full_nextlat | 40.6 +/- 14.2 | 50.9 +/- 6.6 | 10.3 +/- 7.6 | 34.4 +/- 19.6 | 18.4 +/- 11.0 | 2/3 |
| dense_response_gpt | 60.3 +/- 12.8 | 60.2 +/- 12.6 | -0.1 +/- 0.2 | 11.5 +/- 19.8 | 11.5 +/- 19.8 | 3/3 |
| dense_response_nextlat | 79.9 +/- 4.5 | 80.1 +/- 4.4 | 0.1 +/- 0.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 3/3 |
| sparse_full_gpt | 22.3 +/- 20.9 | 22.2 +/- 20.8 | -0.1 +/- 0.1 | 61.9 +/- 34.8 | 61.8 +/- 34.9 | 3/3 |
| sparse_full_nextlat | 9.2 +/- 8.1 | 18.1 +/- 15.7 | 8.9 +/- 7.7 | 88.0 +/- 10.7 | 72.9 +/- 26.6 | 3/3 |
| sparse_response_gpt | 61.6 +/- 9.4 | 61.6 +/- 9.4 | 0.0 +/- 0.0 | 3.1 +/- 5.4 | 3.1 +/- 5.4 | 3/3 |
| sparse_response_nextlat | 77.7 +/- 8.9 | 76.7 +/- 8.5 | -1.0 +/- 0.6 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 3/3 |

The intervention changes only the bias-free 24-class output matrix, initialized from native weights. It fits separate probe families at delays 4,8,16, uses fixed regularization toward the original head, and permits the full vocabulary at test time. The Transformer and at-query hidden states are frozen.

Interpretation: refitting partly improves some full-supervision conditions, but it does not generally recover response-supervised performance. A pure output-head explanation is not established. In dense/full conditions, only one of three matched NTP/NextLat pairs passes the original training gate, so incomplete learning remains a confound. Failed refitting is not proof of erasure; the fit is trained at short delays and may itself fail under length shift.

At-query states have already used full attention. These results do not by themselves establish memory in a pre-query state. All 24 runs and all evaluated delays are retained; no seed, checkpoint or optimizer setting was selected using these test scores. The protocol and exact fitting source hash were recorded before evaluation.
