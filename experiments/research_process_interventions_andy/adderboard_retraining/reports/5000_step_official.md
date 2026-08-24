# Official AdderBoard Evaluation After 5,000-Step Retraining

## Executive summary

All 18 unique architecture candidates representing the 24 research trajectories were retrained successfully on Modal for 5,000 optimizer steps (2,560,000 generated training examples per model). Each saved best checkpoint was then evaluated by the vendored AdderBoard `verify.py` loop without modifying its cases, order, comparison, seed, or qualification rule.

- Completed training jobs: **18/18**
- Completed official evaluations: **18/18**
- Modal/runtime exceptions: **0**
- Official test calls per model: **10,010**
- AdderBoard-qualified models (at least 99%): **0/18**
- Best model: **`8342dd53bf2a-p5520`** with **236/10,010 = 2.3576%**
- Median model: **1/10,010 = 0.0100%**
- Parameter range: **19–5,520**

The principal result is negative but clear: five times more optimization than the earlier 1,000-step screen improved token-level behavior, but almost none of the compact architectures learned exact 10-digit addition. The 5,520-parameter model is the sole meaningful outlier and is the only candidate worth advancing to a substantially longer training study based on these results.

## What “official AdderBoard evaluation” means here

The evaluation calls the repository's vendored `verify.py` directly. The candidate wrapper only implements the submission interface required by that file: `build_model()` and `add(model, a, b) -> int`.

| Protocol property | Value |
|---|---:|
| Verifier source | `vendor/AdderBoard/verify.py` |
| Verifier SHA-256 | `1135b7cd1e335a3b50121cc9ebe68fdacafa235ba82e1d4d65554408db009ec9` |
| Fixed edge-case calls | 10 |
| Random calls | 10,000 |
| Total calls | 10,010 |
| Random generator | `random.Random(2025)` |
| Comparison | Returned integer equals `a + b` |
| Qualification threshold | Accuracy at least 99% |
| Verifier loop modified | No |

This intentionally preserves the verifier's duplicated `(9_999_999_999, 9_999_999_999)` edge case. Evaluation is serial, one `add()` invocation at a time, just as in AdderBoard. The retraining schedule itself is a standalone exploratory 5,000-step schedule, not a claim that the architectures were trained with any particular leaderboard entrant's private training recipe.

## Training configuration

| Setting | Value |
|---|---:|
| Cohort | `adderboard-official-develop-v1` |
| Seed | 1 |
| Optimizer steps | 5,000 |
| Global batch size | 512 |
| Examples processed per model | 2,560,000 |
| Optimizer | AdamW |
| Peak learning rate | 0.001 |
| Warmup | 300 steps |
| Schedule | Cosine decay to zero |
| Validation | 1,000 examples every 500 steps |
| Checkpoints | Every 500 steps |
| Hardware | One Modal T4 per job, at most 8 concurrent jobs |
| Numeric mode | Float32, deterministic CUDA algorithms |

Across the 18 models, training took 2,272.9 aggregate GPU-seconds and official verification took 3,260.3 aggregate GPU-seconds. Mean per-model times were 126.3 seconds for training and 181.1 seconds for official verification.

## Results by unique architecture

“AR token” is the fraction of answer tokens produced correctly by greedy autoregressive generation on the 1,000-case development set. “Dense exact” requires the whole generated answer sequence to match. It is diagnostic only; the official AdderBoard column is the benchmark result.

| Candidate | Params | Traj. | Official passed | Official accuracy | Qualified | AR token | Dense exact | Best step | Train / verify (s) |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| `8342dd53bf2a-p5520` | 5,520 | 1 | 236 / 10,010 | **2.3576%** | No | **78.30%** | **2.80%** | 5,000 | 156.9 / 341.5 |
| `4e8ab5b2829f-p1328` | 1,328 | 1 | 2 / 10,010 | 0.0200% | No | 18.80% | 0.00% | 5,000 | 118.4 / 137.5 |
| `19933b6dca67-p2384` | 2,384 | 2 | 2 / 10,010 | 0.0200% | No | 20.37% | 0.00% | 5,000 | 131.8 / 232.5 |
| `9474e1d8188d-p19` | 19 | 1 | 1 / 10,010 | 0.0100% | No | 12.29% | 0.00% | 5,000 | 106.9 / 126.6 |
| `9a66f924efca-p19` | 19 | 1 | 1 / 10,010 | 0.0100% | No | 11.85% | 0.00% | 5,000 | 109.5 / 126.4 |
| `ebcea5ce8322-p19` | 19 | 1 | 1 / 10,010 | 0.0100% | No | 2.12% | 0.00% | 5,000 | 122.4 / 142.5 |
| `44589818c018-p34` | 34 | 1 | 1 / 10,010 | 0.0100% | No | 12.64% | 0.00% | 4,500 | 133.4 / 161.1 |
| `f4062d778550-p46` | 46 | 2 | 1 / 10,010 | 0.0100% | No | 18.49% | 0.00% | 4,500 | 117.8 / 154.7 |
| `7efe50356b34-p234` | 234 | 1 | 1 / 10,010 | 0.0100% | No | 12.03% | 0.00% | 1,500 | 118.0 / 150.5 |
| `625735ad1857-p376` | 376 | 1 | 1 / 10,010 | 0.0100% | No | 12.59% | 0.00% | 5,000 | 118.0 / 151.4 |
| `7e56691c05a5-p756` | 756 | 1 | 1 / 10,010 | 0.0100% | No | 18.48% | 0.00% | 4,000 | 131.5 / 156.5 |
| `b9ee85f83ffc-p1888` | 1,888 | 2 | 1 / 10,010 | 0.0100% | No | 26.57% | 0.00% | 5,000 | 119.3 / 143.2 |
| `e9d794d5f4a8-p2004` | 2,004 | 1 | 1 / 10,010 | 0.0100% | No | 20.91% | 0.00% | 4,500 | 149.0 / 273.7 |
| `babc7fa2200e-p2384` | 2,384 | 4 | 1 / 10,010 | 0.0100% | No | 19.80% | 0.00% | 5,000 | 131.9 / 219.3 |
| `59fe97a5bdb0-p3440` | 3,440 | 1 | 1 / 10,010 | 0.0100% | No | 19.01% | 0.00% | 5,000 | 148.7 / 305.2 |
| `a36e2bcb9a6c-p46` | 46 | 1 | 0 / 10,010 | 0.0000% | No | 12.31% | 0.00% | 2,500 | 107.8 / 126.9 |
| `df59161bf0d5-p376` | 376 | 1 | 0 / 10,010 | 0.0000% | No | 21.07% | 0.00% | 3,000 | 115.3 / 132.7 |
| `26fd6c2d26b8-p756` | 756 | 1 | 0 / 10,010 | 0.0000% | No | 19.90% | 0.00% | 5,000 | 136.2 / 178.2 |

## Results mapped back to all 24 trajectories

Some trajectories converged to identical architecture artifacts, so 24 trajectories correspond to 18 unique trained models.

| Trajectory | Candidate | Params | Official result |
|---|---|---:|---:|
| `ar-size-h12-20260822-b000-rd0-s1` | `7efe50356b34-p234` | 234 | 1 / 10,010 (0.0100%) |
| `ar-size-h12-20260822-b000-rd1-s1` | `26fd6c2d26b8-p756` | 756 | 0 / 10,010 (0.0000%) |
| `ar-size-h12-20260822-b000-rd2-s1` | `9474e1d8188d-p19` | 19 | 1 / 10,010 (0.0100%) |
| `ar-size-h12-20260822-b000-rd3-s1-r5` | `625735ad1857-p376` | 376 | 1 / 10,010 (0.0100%) |
| `ar-size-h20-20260822-b000-rd0-s1` | `9a66f924efca-p19` | 19 | 1 / 10,010 (0.0100%) |
| `ar-size-h20-20260822-b000-rd1-s1` | `ebcea5ce8322-p19` | 19 | 1 / 10,010 (0.0100%) |
| `ar-size-h20-20260822-b000-rd2-s1` | `f4062d778550-p46` | 46 | 1 / 10,010 (0.0100%) |
| `ar-size-h20-20260822-b000-rd3-s1` | `df59161bf0d5-p376` | 376 | 0 / 10,010 (0.0000%) |
| `ar-size-h8-20260822-b000-rd0-s1` | `f4062d778550-p46` | 46 | 1 / 10,010 (0.0100%) |
| `ar-size-h8-20260822-b000-rd1-s1` | `59fe97a5bdb0-p3440` | 3,440 | 1 / 10,010 (0.0100%) |
| `ar-size-h8-20260822-b000-rd2-s1` | `e9d794d5f4a8-p2004` | 2,004 | 1 / 10,010 (0.0100%) |
| `ar-size-h8-20260822-b000-rd3-s1` | `7e56691c05a5-p756` | 756 | 1 / 10,010 (0.0100%) |
| `oe-size-h12-20260822-b000-rd0-s1` | `a36e2bcb9a6c-p46` | 46 | 0 / 10,010 (0.0000%) |
| `oe-size-h12-20260822-b000-rd1-s1` | `19933b6dca67-p2384` | 2,384 | 2 / 10,010 (0.0200%) |
| `oe-size-h12-20260822-b000-rd2-s1` | `babc7fa2200e-p2384` | 2,384 | 1 / 10,010 (0.0100%) |
| `oe-size-h12-20260822-b000-rd3-s1` | `b9ee85f83ffc-p1888` | 1,888 | 1 / 10,010 (0.0100%) |
| `oe-size-h20-20260822-b000-rd0-s1-r3` | `19933b6dca67-p2384` | 2,384 | 2 / 10,010 (0.0200%) |
| `oe-size-h20-20260822-b000-rd1-s1-r3` | `babc7fa2200e-p2384` | 2,384 | 1 / 10,010 (0.0100%) |
| `oe-size-h20-20260822-b000-rd2-s1-r2` | `babc7fa2200e-p2384` | 2,384 | 1 / 10,010 (0.0100%) |
| `oe-size-h20-20260822-b000-rd3-s1` | `4e8ab5b2829f-p1328` | 1,328 | 2 / 10,010 (0.0200%) |
| `oe-size-h8-20260822-b000-rd0-s1` | `8342dd53bf2a-p5520` | 5,520 | **236 / 10,010 (2.3576%)** |
| `oe-size-h8-20260822-b000-rd1-s1` | `babc7fa2200e-p2384` | 2,384 | 1 / 10,010 (0.0100%) |
| `oe-size-h8-20260822-b000-rd2-s1` | `b9ee85f83ffc-p1888` | 1,888 | 1 / 10,010 (0.0100%) |
| `oe-size-h8-20260822-b000-rd3-s1-r2` | `44589818c018-p34` | 34 | 1 / 10,010 (0.0100%) |

## What changed from the 1,000-step screen

The earlier screen used a 512-case development diagnostic and was not the official leaderboard verifier. It remains useful only for comparing learning progress.

| Diagnostic | 1,000 steps | 5,000 steps |
|---|---:|---:|
| Mean teacher-forced token accuracy | 14.60% | 20.76% |
| Mean autoregressive token accuracy | 15.18% | 19.86% |
| Models with improved teacher-forced token accuracy | — | 13 / 18 |
| Models with improved autoregressive token accuracy | — | 11 / 18 |

Longer training improved partial digit prediction, especially for `8342dd53bf2a-p5520`, but partial token accuracy is not addition accuracy. A single wrong digit makes the returned integer incorrect. This explains why models with roughly 20% token accuracy still score approximately zero under AdderBoard.

## Interpretation

1. **The 5,520-parameter architecture is qualitatively different.** It reached 78.3% autoregressive token accuracy, 2.8% dense exact sequence accuracy, and 2.3576% official accuracy. These three measures agree that it learned some genuine addition behavior rather than merely exploiting one or two trivial cases.

2. **Most tiny architectures appear capacity- or optimization-limited at this budget.** Seventeen models produced at most two exact official sums. Their token accuracies are generally far from the regime where exact 11-digit outputs become common.

3. **The benchmark is doing what it should.** The official integer-equality score exposes the difference between local token competence and complete arithmetic correctness. Reporting token accuracy alone would substantially overstate performance.

4. **Parameter minimization cannot yet be assessed among qualified models.** The leaderboard objective is smallest parameter count subject to at least 99% accuracy. Since no candidate satisfies the constraint, the current results identify a training candidate, not a valid low-parameter leaderboard entry.

5. **Recommended next run:** advance only `8342dd53bf2a-p5520` to a longer schedule first (for example 30,000 steps with intermediate official evaluations or a separate fixed validation set). Running every sub-2,500-parameter candidate much longer is low priority until there is evidence their exact-match curves have left zero.

## Local artifacts and integrity checks

All artifacts are under:

`downloads/adderboard-official-develop-v1/<candidate>/develop-seed-1/`

Each candidate directory contains:

- `best_checkpoint.pt`
- `standalone_summary.json`
- `training_events.jsonl` (exactly 5,000 records)
- `official_adderboard_verify.log`
- trainer-generated configuration and validation artifacts

Local validation passed for all 18 candidates:

- checkpoint SHA-256 equals the digest recorded in the summary;
- training reports success, 5,000 completed steps, and 2,560,000 examples;
- official evaluation reports 10,010 cases, seed 2025, and an unmodified verifier loop;
- all event and verifier logs are present and nonempty.

The complete downloaded cohort is approximately 57 MB. The implementation and these results remain in the standalone directory and did not modify either Git repository.
