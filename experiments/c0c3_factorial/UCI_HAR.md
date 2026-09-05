# UCI HAR Pareto campaign — prepared, not launched

This is a five-block Greedy OpenEvolve v2.1 campaign with C0–C3 only, 20 runs,
200 proposals per run and 4,000 proposals in total. There is no C4 or N0.
The Codex preset is GPT-5.6 Sol, xhigh reasoning, fast service tier, with a
fresh subject session per proposal. C1/C3 receive the same assumption-changing
instruction at proposals 10, 20, …, 200. C0/C2 receive ordinary instructions.

## Objectives and retention

The two objectives are validation **accuracy** (maximize) and executed affine
inference **MACs per example** (minimize). Macro-F1, cross-entropy and parameters
are supplementary metrics. There is no minimum accuracy qualification gate.

Every valid evaluated candidate, including the seed and non-retained candidates,
remains in a private run archive. A candidate is admitted if its objective pair
is newly nondominated against that entire archive. Exact objective ties keep
the earlier candidate. Invalid evaluations never enter the frontier.

C0/C1 expose only the current incumbent. Any admitted child replaces it,
including an incomparable child with lower accuracy and lower MACs. C2/C3 expose
at most four models: admitted children fill vacant slots, then replace only
their selected parent. Parent selection is least-selected retained lineage,
then earliest retention order, then candidate ID; a replacement inherits its
parent's selection count. No scalar accuracy or hypervolume ranks parents.

The visible portfolio is bounded research memory, **not the entire frontier**;
it can contain a design later dominated by another lineage. Rejected and evicted
source is never revived. The private archive is used identically in all four
conditions for admission and reporting; it is not extra subject-visible memory.
This is a prospective multi-objective retention variant of v2.1, not the legacy
strict-scalar-improvement experiment. It must be reported as a separate stratum.

## Hypervolume and dashboard

The reference is fixed at **accuracy 0 and 2,000,000 MACs**. The axes are linear.
The hypervolume is the union of rectangles between each archive-frontier point
and that reference, divided by 2,000,000. It lies in [0,1], is higher-is-better,
and cannot decrease as a run accumulates observations. No normalization is
recomputed using later observations or another run's results.

The reference is an explicit modeling choice, not a claim that hypervolume is
preference-free. The 2,000,000-MAC ceiling also defines evaluator admissibility;
valid candidates cannot lie beyond the plotted cost range. Boundary points can
join the frontier without increasing its area. Accuracy zero is permitted.

The dashboard's primary trajectory is archive hypervolume, not an individual
model score. Accuracy and MACs remain available as separate metric axes; choosing
MACs for x and accuracy for y displays candidate tradeoffs. The evaluator's
legacy `fitness` compatibility field contains accuracy but never drives Pareto
admission, parent selection or the primary chart. The incumbent means the last
admitted visible design, not a claimed unique best model.

Layer C retrains and evaluates every member of the final validation archive
frontier, including evicted designs, on the sealed official test split. It reports
the resulting test hypervolume, without choosing models using test results.
If any frontier evaluation fails, the run's test hypervolume is marked incomplete.
This additional post-search cost depends on frontier size and is outside the
4,000 proposal-evaluation budget.

## Seed and data

The seed is MicroBiConvLSTM from commit
`7d0bf16219865123584059bfc1ccda532170d3fc` of
[the author's repository](https://github.com/WhiteMetagross/MicroBiConvLSTM).
Constructor, initialization and forward computation are preserved, with formatting
and typing cleanup and a thin evaluator training interface. It has 10,454
parameters. Training uses the upstream UCI hyperparameters: batch 64, AdamW,
learning rate 0.002185, weight decay 0.000142, dropout 0.15, and gradient clip 1.
The cosine schedule spans the fixed training exposure instead of the paper's
200-epoch/early-stopping schedule. No pretrained weights are inherited.
The reported paper macro-F1 of 93.41% is not a reproduced result for this setup.
File hashes and adaptation details are in `uci_har_seed_provenance.json`.

Preparation calibration on Modal measured 91.79% accuracy, 92.05% macro-F1 and
48.3 evaluator seconds for the final source. An earlier Modal calibration measured
90.70% accuracy in 34.5 seconds; the Windows check measured 90.05%. These use the
same numeric seed and equivalent model computation, but CPU/library differences
can alter training results. Deterministic flags do not guarantee equality across
machines. Preserve the first final-source calibration as the baseline; do not
choose the best score among checks. A later deployment verification is diagnostic.

The upstream dataset wrapper imports `nanoharmamba`, which is absent from that
repository. The protected evaluator therefore reads the public UCI archive
directly after verifying SHA-256
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
It uses nine inertial channels and 128 samples per window, in `[batch,time,channel]`
order. Channel normalization is fitted only to the training people.

* Training: 5,975 windows from 17 official training people.
* Validation: 1,377 windows from people 5, 14, 19 and 26.
* Holdout: the official 2,947-window, nine-person test split, unopened by Layer A.

Each evaluation uses exactly 50,000 presented training examples, fresh weights,
at most 100,000 parameters, and a 240-second process timeout. Evaluation queueing
does not consume this timeout. Each run has a 48,000-second evaluator budget
(200 times the per-call ceiling) and the existing 100M-token accounting ceiling.

MACs count executed Linear, Conv1d/2d and RNN/GRU/LSTM affine operations,
including all layers/directions and their cell variants. Bias, normalization,
activations, pooling and elementwise recurrent gates are excluded. Unsupported
affine primitives fail closed. This is a reproducible computational proxy, not
measured device latency or joules. The seed uses **420,128 MACs** under this rule.

## Execution and hold

Modal app: `rl4rl-uci-har-v21`. Each evaluator requests **2 CPU cores, 4,096 MiB
RAM and no GPU**, exactly matching Tiny AdderBoard's allocation. There is no
application-level Modal container cap and no Codex-worker cap. The 20 sequential
run controllers bound this campaign's natural simultaneous evaluations to 20;
account-level provider limits still apply. Matching per-worker allocation does
not promise an identical total bill. Actual worker seconds are logged.

Every new evaluation tries Modal first. Infrastructure failure permits Windows
fallback; candidate/model failures do not. Uncertain remote calls must be
cancelled before fallback. Queued fallback work periodically retries Modal.
Windows fallback and ordinary Windows CommandEvaluator calls share one hard
three-slot host pool across campaigns and detached checkouts. Leases release
on process exit. The pool is under `shared_local_evaluator_root()` in
`windows-three-slot-fallback`, not inside an individual campaign.

Existing Python processes that imported an older evaluator module retain their
old scheduler until their next normal restart. No existing campaigns are paused,
restarted, or amended by this preparation. Do not claim old workers have adopted
the new pool before verifying their runtime version at launch review.

The campaign path is `data/c0c3/uci-har-pareto-v21`. Preparation deploys the
evaluator and calibrates only the seed. It must leave all runs ready with zero
proposals, no active opportunities and no launch receipt. There is no automation.

After explicit operator review, the launch entry point is
`experiments/c0c3_factorial/start_uci_har_v21.ps1 -Approved`. Without `-Approved`
it refuses to start. It validates the 20-run roster, Pareto settings and all
campaign hashes before creating hidden Windows workers. Do not execute this
command during preparation. A source change after campaign creation requires
the ordinary provenance/hash validation workflow before launch.
