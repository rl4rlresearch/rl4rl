# AdderBoard Autoresearch pilot

You are running a bounded, autonomous research pilot. Your job is to search
for a **smaller trained autoregressive transformer** for 10-digit addition.

## Objective

Minimize the official AdderBoard unique parameter count, subject to at least
99% accuracy on the official verifier (10 edge cases plus 10,000 fixed-seed
random cases, seed 2025). A candidate is retained only when it qualifies and
uses *strictly fewer* parameters than the current retained incumbent.

The starting incumbent is a 6,080-parameter, 100%-verified conventional
decoder-only transformer. Do not assume its architecture is close to globally
optimal. Continue testing substitution-level changes, not only deletions and
width reductions. Never write a conclusion that a local floor is a global
lower bound.

## Scope and anti-leakage boundary

Work only in this workspace. Do not read the parent RL4RL repository, online
leaderboards, other submissions, or external source code. You may use your own
general technical knowledge, but the experiment should not be handed a known
solution or a target parameter count.

You may edit only:

- `src/model.py`
- `src/data.py`
- `src/train.py`
- `submission.py`

Do not edit `PROGRAM.md`, the official verifier, the runner, `RUN_CONFIG.json`,
or any archived attempt artifacts. Keep the public contracts intact:

- `submission.py` must provide `build_model()` and `add(model, a, b)`;
- the model must remain a tensor-in/logits-out autoregressive transformer with
  at least one self-attention layer;
- `add` must remain generic greedy decoding, with no addition-specific Python
  arithmetic or carry logic;
- the reported parameter count must be calculated from actual, deduplicated
  model parameters. Never manually claim a smaller number.

## Required experiment loop

1. Inspect the retained source and `../RESULTS.tsv`.
2. Write a concise mechanism hypothesis and proposal in the `--proposal` and
   `--description` arguments below.
3. Make one coherent candidate change. A change may be an ablation, but include
   representational substitutions as well as local compression ideas.
4. Run exactly one logged attempt:

   ```bash
   python ../run_attempt.py --run-dir .. attempt --description "short factual description" --proposal "mechanism hypothesis and what changed"
   ```

   The runner trains the current candidate, invokes the untouched official
   verifier, saves code/checkpoint/stdout/stderr before any rollback, appends a
   TSV row, creates a permanent Git ref, and restores the retained incumbent if
   the candidate is not accepted.
5. Read the resulting evidence. Do not relabel a failure as a success and do
   not change the retention rule.
6. Continue until the runner reports that the fixed attempt budget is exhausted,
   then stop and summarize the empirical results without claiming a global
   optimum.

The baseline must be recorded before the first attempt. If it is not already
in `../RESULTS.tsv`, run:

```bash
python ../run_attempt.py --run-dir .. baseline
```

## What to preserve in reasoning

For each proposal, distinguish a parameterization-preserving compression from
a representational change (for example, token representation, positional
integration, deterministic/tied projections, attention organization, or
feed-forward mechanism). The source snapshots and Codex event log are research
artifacts, not scratch files.
