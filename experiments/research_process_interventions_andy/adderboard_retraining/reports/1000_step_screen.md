# AdderBoard smallest-architecture training screen

## Executive summary

All 18 structurally unique final architectures from the 24 research
trajectories completed a clean 1,000-step Modal T4 training run. Every run used
the same seed, optimizer, batch size, data generator, and evaluation set.

- Successful runs: **18/18**
- Optimizer steps: **1,000 per architecture**
- Training examples: **512,000 per architecture; 9,216,000 total**
- Evaluation cases: **512 per architecture**
- Full-answer exact match: **0.00% for every architecture**
- Best autoregressive token accuracy: **21.34%**
- Best teacher-forced token accuracy: **20.78%**
- Aggregate recorded training time: **486.1 T4-seconds**
- Raw local artifacts: **144 files, approximately 13 MB**

The main result is not that every architecture learned addition. None produced
a completely correct 11-digit answer plus end token in the 512-case evaluation.
However, several models learned partial token-level structure: the strongest
models generated about 20–21% of answer tokens correctly, compared with a
uniform 15-token baseline of about 6.67%.

This was supervised AdamW training on addition examples. It was not
reinforcement learning of the research-agent policy.

## Experimental setup

| Setting | Value |
|---|---:|
| Cohort | `small-arch-screen-v1-retry2` |
| Unique architectures | 18 |
| Source trajectories represented | 24 |
| Seed | 1 |
| Accelerator | NVIDIA Tesla T4 |
| PyTorch | 2.7.1+cu126 |
| Optimizer | AdamW |
| Peak learning rate | 0.001 |
| Schedule | 100-step warmup, cosine decay to zero |
| Global batch size | 512 |
| Training steps | 1,000 |
| Validation interval | 100 steps |
| Validation cases | 512 |
| Loss | Answer-only cross-entropy |
| Examples seen per model | 512,000 |

Three metrics are reported:

- **Exact match:** the entire autoregressively generated answer must match,
  including every digit and the end token.
- **Autoregressive token accuracy:** the fraction of answer positions correct
  while the model consumes its own earlier generated tokens.
- **Teacher-forced token accuracy:** the fraction correct when the model is
  given the true preceding answer tokens. This is easier and can expose models
  that learn local predictions but cannot roll them out reliably.

## Results for all 18 architectures

Sorted by autoregressive token accuracy.

| Candidate | Parameters | AR token | Teacher-forced | Exact match | Best dev loss | Final train loss | Best step | Trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `8342dd53bf2a-p5520` | 5,520 | **21.34%** | **20.78%** | 0.00% | **2.0686** | 1.3849 | 1,000 | 1 |
| `babc7fa2200e-p2384` | 2,384 | **21.16%** | 16.78% | 0.00% | 2.4736 | 1.5076 | 1,000 | 4 |
| `b9ee85f83ffc-p1888` | 1,888 | **20.70%** | **20.26%** | 0.00% | 2.2114 | 1.5273 | 1,000 | 2 |
| `7e56691c05a5-p756` | 756 | **20.43%** | 16.00% | 0.00% | 2.4764 | 1.5878 | 1,000 | 1 |
| `26fd6c2d26b8-p756` | 756 | **20.23%** | 15.95% | 0.00% | 2.5855 | 1.6227 | 1,000 | 1 |
| `4e8ab5b2829f-p1328` | 1,328 | 19.84% | 16.05% | 0.00% | 2.5334 | 1.5358 | 300 | 1 |
| `19933b6dca67-p2384` | 2,384 | 19.65% | 16.46% | 0.00% | 2.5238 | 1.5186 | 300 | 2 |
| `e9d794d5f4a8-p2004` | 2,004 | 18.88% | 15.93% | 0.00% | 2.4830 | 1.5024 | 1,000 | 1 |
| `59fe97a5bdb0-p3440` | 3,440 | 18.64% | 16.29% | 0.00% | 2.4249 | 1.4645 | 900 | 1 |
| `9a66f924efca-p19` | 19 | 12.30% | 11.25% | 0.00% | 2.7614 | 2.5642 | 1,000 | 1 |
| `ebcea5ce8322-p19` | 19 | 12.30% | 11.87% | 0.00% | 2.8295 | 2.5454 | 1,000 | 1 |
| `44589818c018-p34` | 34 | 12.30% | 9.46% | 0.00% | 3.3062 | 2.8274 | 1,000 | 1 |
| `7efe50356b34-p234` | 234 | 12.30% | 9.57% | 0.00% | 2.6366 | 1.9955 | 1,000 | 1 |
| `625735ad1857-p376` | 376 | 12.13% | 12.11% | 0.00% | 2.5864 | 2.0160 | 1,000 | 1 |
| `df59161bf0d5-p376` | 376 | 12.00% | 13.72% | 0.00% | 3.0396 | 2.1854 | 1,000 | 1 |
| `9474e1d8188d-p19` | 19 | 9.28% | 12.16% | 0.00% | 4.0896 | 3.6889 | 1,000 | 1 |
| `f4062d778550-p46` | 46 | 7.05% | 17.45% | 0.00% | 3.1430 | 2.9661 | 1,000 | 2 |
| `a36e2bcb9a6c-p46` | 46 | 2.72% | 10.64% | 0.00% | 3.0188 | 2.8978 | 1,000 | 1 |

## What the screen suggests

### 1. The extreme parameter minima do not look competitive

The 19–46 parameter candidates reached only 2.72–12.30% autoregressive token
accuracy. Their losses were also generally worse. These candidates had won the
architecture searches because zero exact-match accuracy was an eligible floor
and parameter minimization could dominate. Longer training may improve them,
but this screen gives no evidence that they are the best use of the next compute
budget.

### 2. A meaningful capacity transition appears near 756 parameters

All nine models with 756–5,520 parameters achieved 18.64–21.34%
autoregressive token accuracy. None of the 19–376 parameter models exceeded
12.30%. This is only a single-seed screen, but it is a much clearer capacity
signal than the original 10-step exact-match evaluations.

Across these 18 points, log10 parameter count correlates positively with
autoregressive token accuracy (**Pearson r = 0.827**) and negatively with best
development loss (**r = -0.780**). These are descriptive associations, not
causal estimates; architecture family and parameter count vary together.

### 3. Teacher-forcing can hide rollout failure

The 46-parameter `f4062d778550-p46` model reached 17.45% teacher-forced token
accuracy but only 7.05% autoregressive token accuracy. It can make some local
predictions when supplied the correct history, but its own mistakes compound
during generation. This is exactly why exact match and autoregressive token
accuracy should remain primary metrics.

### 4. Zero exact match does not mean zero learning

Exact match requires 12 successive output positions to be correct. A model with
about 20% token accuracy can still have essentially zero probability of a fully
correct sequence if errors are widespread. The dense metrics show partial
learning that the earlier `0.0000` exact-match display could not reveal.

## Recommended 5,000-step development set

Advance the nine candidates with at least 18% autoregressive token accuracy and
best development loss below 2.60:

1. `8342dd53bf2a-p5520`
2. `babc7fa2200e-p2384`
3. `b9ee85f83ffc-p1888`
4. `7e56691c05a5-p756`
5. `26fd6c2d26b8-p756`
6. `4e8ab5b2829f-p1328`
7. `19933b6dca67-p2384`
8. `e9d794d5f4a8-p2004`
9. `59fe97a5bdb0-p3440`

Recommended next design: **5,000 steps × 3 seeds** for these nine candidates.
Use the mean and spread of autoregressive token accuracy, exact match, and
development loss. Only models showing consistent progress should receive the
full 30,000-step treatment.

## Candidate-to-trajectory mapping

| Candidate | Architecture | Source trajectories |
|---|---|---|
| `19933b6dca67-p2384` | Attention-Only Multidepth Decoder | `oe-size-h12-20260822-b000-rd1-s1`; `oe-size-h20-20260822-b000-rd0-s1-r3` |
| `26fd6c2d26b8-p756` | Width Twelve Routing Capacity Test | `ar-size-h12-20260822-b000-rd1-s1` |
| `44589818c018-p34` | Scalar Affine Residual Attention Adder | `oe-size-h8-20260822-b000-rd3-s1-r2` |
| `4e8ab5b2829f-p1328` | Single-Attention Carry Decoder | `oe-size-h20-20260822-b000-rd3-s1` |
| `59fe97a5bdb0-p3440` | Staged Attention-Only Decoder | `ar-size-h8-20260822-b000-rd1-s1` |
| `625735ad1857-p376` | Eight-channel attention-only residual model | `ar-size-h12-20260822-b000-rd3-s1-r5` |
| `7e56691c05a5-p756` | Width-12 Single-Attention Addition Decoder | `ar-size-h8-20260822-b000-rd3-s1` |
| `7efe50356b34-p234` | Six-Channel Three-Head RMS Decoder | `ar-size-h12-20260822-b000-rd0-s1` |
| `8342dd53bf2a-p5520` | Sinusoidal AdderBoard Decoder | `oe-size-h8-20260822-b000-rd0-s1` |
| `9474e1d8188d-p19` | Scalar Residual Dual-Position Attention Decoder | `ar-size-h12-20260822-b000-rd2-s1` |
| `9a66f924efca-p19` | Scalar Pure Context Attention Addition Model | `ar-size-h20-20260822-b000-rd0-s1` |
| `a36e2bcb9a6c-p46` | Compact Residual Attention AdderBoard Decoder | `oe-size-h12-20260822-b000-rd0-s1` |
| `b9ee85f83ffc-p1888` | Attention-Only Residual AdderBoard Decoder | `oe-size-h8-20260822-b000-rd2-s1`; `oe-size-h12-20260822-b000-rd3-s1` |
| `babc7fa2200e-p2384` | Sinusoidal Attention-Only AdderBoard Decoder | `oe-size-h8-20260822-b000-rd1-s1`; `oe-size-h12-20260822-b000-rd2-s1`; `oe-size-h20-20260822-b000-rd1-s1-r3`; `oe-size-h20-20260822-b000-rd2-s1-r2` |
| `df59161bf0d5-p376` | Single-Head Bottleneck Addition Decoder | `ar-size-h20-20260822-b000-rd3-s1` |
| `e9d794d5f4a8-p2004` | Serial Operand Refinement Adder Decoder | `ar-size-h8-20260822-b000-rd2-s1` |
| `ebcea5ce8322-p19` | Single-Channel Attention-Dominant Mix | `ar-size-h20-20260822-b000-rd1-s1` |
| `f4062d778550-p46` | Two-channel causal attention-only decoder | `ar-size-h8-20260822-b000-rd0-s1`; `ar-size-h20-20260822-b000-rd2-s1` |

## Artifact locations and validation

- Reproducible scripts and reports: this `adderboard_retraining/` directory.
- Candidate/trajectory manifest: `candidates/manifest.json`.
- Raw downloaded results: the operator's untracked Modal download directory for
  cohort `small-arch-screen-v1-retry2`.

Each candidate directory contains:

- the immutable candidate graph;
- the best and latest checkpoints;
- a partial-resume checkpoint;
- all 1,000 step-level training events;
- the training manifest and training summary;
- the standalone summary with dense accuracy metrics.

Local validation confirmed **18/18 candidate directories**, **144/144 required
files**, **18,000 event records**, matching best-checkpoint SHA-256 digests, and
1,000 completed optimizer steps for every candidate.

## Execution note

Two infrastructure-only attempts preceded the successful retry. The first
failed during remote import because local Mac paths were validated inside the
container. The second used fire-and-forget calls whose workers were interrupted
when the ephemeral app exited. Neither is included in the scientific table.
The final `retry2` cohort used one attached parallel Modal map and completed all
18 runs cleanly.
