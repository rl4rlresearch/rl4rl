# AdderBoard Autoresearch pilot

You are running a bounded, autonomous research pilot. Your job is to search
for the **smallest trained autoregressive transformer** for 10-digit addition.

## Objective

Minimize the official AdderBoard unique parameter count, subject to at least
99% accuracy on the official verifier (10 edge cases plus 10,000 fixed-seed
random cases, seed 2025). A candidate is retained only when it qualifies and
uses *strictly fewer* parameters than the current retained incumbent.

The starting incumbent is a 6,080-parameter, 100%-verified conventional
decoder-only transformer. Do not assume its architecture is close to globally
optimal. Continue testing substitution-level major architecture changes, not only deletions and
width reductions. Never write a conclusion that a local floor is a global
lower bound.

## Scope and anti-leakage boundary

Work only in this workspace. Do not read the parent RL4RL repository, anything online, or external source code. You may use your own
general technical knowledge, but the experiment should not be handed a known
solution or a target parameter count.

You may edit only:

- `src/model.py`
- `src/data.py`
- `src/train.py`
- `submission.py`
- files under `automations/`, solely to implement bounded automations described
  below

Do not edit `PROGRAM.md`, the official verifier, the runner, `RUN_CONFIG.json`,
or any archived attempt artifacts. Keep the public contracts intact:

- `submission.py` must provide `build_model()` and `add(model, a, b)`;
- the model must remain a tensor-in/logits-out autoregressive transformer with
  at least one self-attention layer;
- `add` must remain generic greedy decoding, with no addition-specific Python
  arithmetic or carry logic;
- the reported parameter count must be calculated from actual, deduplicated
  model parameters. Never manually claim a smaller number.

## Evaluation discipline

During the search, the runner's fixed seed-2025 verifier is a development and
qualification signal, not evidence of final generalization. Do not run it
outside the runner, generate extra test sets to select candidates, or claim
that a retained candidate has independently generalized. A separate final
holdout evaluation is performed only after this run has ended; do not inspect,
generate, or optimize against that final holdout during the search.

## Evidence review before a proposal

Before each regular attempt or automation, inspect the retained source and
`../RESULTS.tsv`, then include all of the following in the `--proposal` text:

1. the current retained parameter count and accuracy margin over 99%;
2. the most recent accepted and failed result in the same mechanism family;
3. the number of prior attempts in that family; and
4. why this proposal is more informative than the nearest untested alternative.

Use one of these mechanism-family labels: `feed-forward width`, `token
representation`, `position representation`, `attention organization`,
`normalization`, `parameter tying`, `scalar pruning`, or `training control`.
The proposal must state its label. A failure is useful evidence; do not repeat
it without explaining what has changed in the hypothesis or conditions.

## Required experiment loop

1. Complete the evidence review and write a concise mechanism hypothesis in
   the `--proposal` and `--description` arguments.
2. Make coherent candidate change(s). A change may include ablations,
   representational substitutions, and local compression ideas.
3. For a regular candidate, run exactly one logged attempt:

   ```bash
   python ../run_attempt.py --run-dir .. attempt --description "short factual description" --proposal "mechanism hypothesis and what changed"
   ```

   The runner trains the current candidate, invokes the untouched official
   verifier, saves code/checkpoint/stdout/stderr before any rollback, appends a
   TSV row, creates a permanent Git ref, and restores the retained incumbent if
   the candidate is not accepted.
4. Read the resulting evidence. Do not relabel a failure as a success and do
   not change the retention rule.
5. Continue until the runner reports that the fixed attempt budget is exhausted,
   then stop and summarize the empirical results without claiming a global
   optimum.

The baseline must be recorded before the first attempt. If it is not already
in `../RESULTS.tsv`, run:

```bash
python ../run_attempt.py --run-dir .. baseline
```

## Bounded automations

An automation is a small program that executes a repeated, structured
search policy without requiring a separate agent deliberation for every tiny
variant. Use one when testing a monotonic or repeatedly structured mechanism. The agent's
choice of mechanism and search policy is a **macro-attempt**; each candidate
that the program evaluates is a **micro-trial**.

You may create a helper under `automations/` for this purpose. Before launching it,
start the macro-attempt with:

```bash
python ../run_attempt.py --run-dir .. automation-start --automation-id "short-name" --family "scalar pruning" --description "what the automation tests" --proposal "hypothesis, ordering, acceptance, stopping, and budget" --max-micro-trials 20
```

Then have the helper modify one candidate at a time and invoke:

```bash
python ../run_attempt.py --run-dir .. automation-attempt --description "one micro-trial" --proposal "current automation decision"
```

After it reaches its declared boundary, close the macro-attempt with:

```bash
python ../run_attempt.py --run-dir .. automation-end --summary "attempted range, frontier, failures, compute used, and stop reason"
```

Record in the macro proposal:

- the mechanism hypothesis and family label;
- the candidate-ordering rule (for example, an importance ranking);
- the acceptance/rollback rule;
- the stopping rule;
- a maximum micro-trial count; and
- a training or wall-clock compute budget.

The automation must modify one candidate at a time and invoke
`automation-attempt` once
for every micro-trial. It must never train, verify, retain, or silently discard
candidates outside `run_attempt.py`. `AUTOMATION_RESULTS.tsv` records every
micro-trial; `RESULTS.tsv` receives one summary row when `automation-end` closes
the macro-attempt. After each micro-trial, the helper may read
`../AUTOMATION_RESULTS.tsv`
and `../STATE.json` to decide its next eligible candidate. Stop the automation at its declared boundary, then write the required
`automation-end` summary. Only then should you select a new mechanism or revise
the automation.

## Interpreting errors

When an attempt has status `error`, inspect its train and verifier logs before
making another proposal. Classify it in the next proposal as one of:

- `infrastructure`: command, timeout, dependency, or filesystem problem;
- `implementation`: broken model, data, or submission contract;
- `optimization failure`: the candidate trained but never reached the
  checkpoint/validation criterion; or
- `nonqualification`: a valid checkpoint received a score below threshold.

An error without an official score is not automatically evidence that the
architecture lacks capacity. You may repair one reproducible infrastructure or
implementation error for the same intended candidate, but must identify it as
a repair. Do not silently retry an optimization failure; explain why changed
conditions justify it.

## What to preserve in reasoning

For each proposal, distinguish a parameterization-preserving compression from
a representational change (for example, token representation, positional
integration, deterministic/tied projections, attention organization, or
feed-forward mechanism). The source snapshots and Codex event log are research
artifacts, not scratch files.

## Final report only

Only after the configured attempt budget is exhausted, provide a trajectory
summary in the final response. Do not write this full summary after each
attempt. State the initial and final frontier, number of regular attempts and
micro-trials, allocation across mechanism families, accepted/error/
nonqualification outcomes, important failure boundaries, and the training
steps or wall-clock cost where the logs provide them. State that independent
final-holdout evaluation remains required, and do not claim a global optimum.
