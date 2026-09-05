# Compact Tiny AdderBoard seed

The seed has **1,012 independent trainable parameters** and a reported budget of
**400 optimizer steps**. The unchanged protected verifier confirmed more than
99% exact-answer accuracy on both public validation and sealed holdout for all
four tested fresh initializations. Verification was performed on September 5,
2026 on the Windows i7-1255U host using PyTorch 2.14.0+cpu and two CPU threads.

The source is `task_sources/tiny_adderboard_compact/train.py`. The Tiny AdderBoard
task configuration now selects it for future campaigns. The original larger
source remains in `task_sources/tiny_adderboard`; existing campaign snapshots and
records were not edited. No campaign was launched as part of this seed search.

## Verified result

Every row below trained from a fresh initialization for exactly **400 steps**,
with batch size 2,048. Each public and holdout score uses 10,000 unique operand
pairs from the verifier's original disjoint hash partitions.

| Random seed | Public exact accuracy | Holdout exact accuracy | Attention ablated |
| --- | ---: | ---: | ---: |
| 20260828 | 99.80% | 99.84% | 0.00% |
| 1 | 100.00% | 100.00% | 0.00% |
| 42 | 100.00% | 100.00% | 0.00% |
| 2026 | 99.62% | 99.66% | 0.00% |

Attention ablation is the protected verifier's temporary removal of learned
attention parameters, evaluated on 1,000 public cases. Those parameters are
restored afterward; the submitted model has no pruned or frozen-zero weights.

The final source was frozen before any holdout result was inspected. Public
development and replication scores informed selection. All four predeclared
holdout runs are reported. These results establish the stated CPU profile;
they do not establish the same convergence for every seed or numerical backend.

## Model and integrity

| Component | Parameters |
| --- | ---: |
| Token embedding and independent projection, rank 3 | 360 |
| Learned absolute positions | 66 |
| Two-head learned relative attention and value/output projections | 106 |
| Feedforward network, width 6 | 84 |
| Three layer normalizations | 36 |
| Independent output projection and classifier, rank 3 | 360 |
| **Total** | **1,012** |

The residual width is six. Input and output matrices are independent, including
their low-rank factors. There are no aliased parameter objects or shared tensor
storage, no frozen model buffers, no reused layer matrices, no answer rules,
and no arithmetic-specific input transformations. Ordinary trainable biases
start at zero; zero initialization does not reduce the parameter count.

The attention probabilities depend on learned relative-position scores, while
the values depend on the input tokens. This is a position-based attention
variant rather than content-based query/key dot-product attention. Every causal
offset receives a learned score; the model is not given digit alignments.
Related general mechanisms are described in
[Synthesizer](https://arxiv.org/abs/2005.00743) and
[Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155).
The implementation is an independently constructed combination of these general
ideas. Arithmetic-specific position coupling, tied weights, fixed arithmetic
features, and pretrained weights from the reference models were not adopted.

Training uses ordinary cross-entropy and AdamW, with learning rates 0.1 for the
relative-position scores and 0.02 for other parameters, weight decay 0.01,
gradient clipping at 1.0, and a cosine schedule. The protected loop performs
exactly one optimizer update per counted step. Data generation, scoring,
decoding, parameter counting, and the evaluator source remain unchanged.

The seed communicates one 400-step budget. Its literal evaluation ladder is
`[400, 1000]`: the latter is the common evaluator fallback and schedule horizon,
not a second advertised seed budget. All reported seed verifications stopped at
the first checkpoint. Earlier experimental timing checkpoints are not included
in the seed's instructions.

## Reproduction on this Windows host

Expect approximately **22 seconds for a complete public evaluator invocation**
on this Windows CPU using two PyTorch threads and one active evaluator. Three
two-thread process measurements had a median of 21.48 seconds. A direct run of
the unchanged evaluator took 19.71 seconds end to end and reported 14.91 seconds
for its training/checkpoint interval. Full process time also includes Python and
PyTorch startup, data preparation, model checks, scoring, and attention ablation.
These figures cover fresh training for 400 steps; they are not inference-only
timings. Other active evaluators or background work can increase the runtime.

Raw timing receipts, including the separate thread-count probes, are in
`outputs/tiny-seed-search/timing-verification/summary.json`. The direct command
receipt is `outputs/tiny-seed-search/direct-evaluator-result.json`, with its wall
time in `direct-evaluator-wall-seconds.txt`. One of the repeated timing runs
overlapped repository tests; the separate direct measurement was sequential.

Use Python with PyTorch installed. The local search environment is
`outputs/tiny-seed-search/venv/Scripts/python.exe`.

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
Remove-Item Env:C0C3_RUN_SEED -ErrorAction SilentlyContinue
& outputs/tiny-seed-search/venv/Scripts/python.exe `
  experiments/c0c3_factorial/tiny_adderboard.py evaluate-ladder `
  --workspace experiments/c0c3_factorial/task_sources/tiny_adderboard_compact `
  --repo-root . --output outputs/compact-seed-public.json `
  --device cpu --seed 20260828 --evaluation-batch-size 2048 `
  --max-steps 1000 --max-parameters 1999
```

Replace `evaluate-ladder` with `holdout` to run the protected final verification,
and use a different output path. Preserve the common schedule horizon shown
above: changing `--max-steps` changes the training procedure, even though the
qualified seed stops at 400 steps.

Thread settings matter for reproducibility. A separate four-thread timing probe
obtained 98.39% public accuracy at 400 steps and needed the fallback. It is not
part of the certified two-thread profile. CUDA and MPS performance were not
measured. The existing August MPS compute-match receipt describes the original
21,952-parameter seed and does not calibrate this replacement.

## Evidence and checks

- Source SHA-256: `1b1effac71d0d87af31bcea443e55a32c7fc7c74fed0cc49e415dd32b4edb937`.
- Unchanged evaluator SHA-256: `f5c10a98f042baedd110edfb6695666ff29fb181634ef9536d736aaf6e1f89f8`.
- Local frozen selection: `outputs/tiny-seed-search/final-selection.json`.
- Full holdout receipts: `outputs/tiny-seed-search/final-verification/summary.json`.
- Development sources, including unsuccessful and stopped trials:
  `outputs/tiny-seed-search/trial-summary.json` and their adjacent logs.
- Package Ruff checks and all nine focused Tiny AdderBoard tests passed.
- Repository-wide tests were attempted, including the runnable tests after
  collection errors. Ten modules cannot collect because Windows lacks `fcntl`;
  four existing dashboard tests also fail in host-capacity/POSIX-locking code.
  Logs are retained under `outputs/tiny-seed-search/repository-pytest*.log`.

The focused tests check source preflight, task selection, parameter independence,
fresh initialization, causal behavior, optimizer coverage, learned routing
gradients, and the boundary between loss evaluation and parameter updates.
Accuracy claims come from full training and the protected evaluator, not those
unit tests. The report and holdout receipts remain outside the seed source
directory so they are not included as subject-visible seed files.
