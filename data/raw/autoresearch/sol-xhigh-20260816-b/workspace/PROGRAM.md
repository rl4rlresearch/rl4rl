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

## Required experiment loop

1. Inspect the retained source and `../RESULTS.tsv`.
2. Write a concise mechanism hypothesis and proposal in the `--proposal` and
   `--description` arguments below.
3. Make one coherent candidate change. A change may be an ablation, but include
   representational substitutions as well as local compression ideas.
4. For a regular candidate, run exactly one logged attempt:

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

## Bounded automations

An automation is a small program that executes a repeated, structured
search policy without requiring a separate agent deliberation for every tiny
variant. Use one when testing a monotonic or repeatedly structured mechanism. 
The agent's choice of mechanism and search policy is a **macro-attempt**; each 
candidate that the program evaluates is a **micro-trial**. If you notice that 
your macro-attempts are becoming repetetive, turn it into an automation and 
wait until the repetetive task does not work anymore, and use your next macro-
attempt on a different task. 

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

Do **not** create a series of automation IDs with
`--max-micro-trials 1` to test a repeated list one candidate at a time. A
one-micro automation is allowed only for a genuinely non-repeatable research
question, and its proposal must explain why no ordered multi-trial helper is
valid. The goal with the micro trials and automations is to save tokens and 
macro-attempts. The maximum micro-trial count, and any other limits on 
automations should allow the automation to keep going until it is exhausted 
rather than hitting a limit early without exhausting itself. Do not keep 
checking on your automations every micro-trial, it is preffered that you to 
trust the automation that you ran will work and set triggers to let you know 
of any problems or when the automation is completed. It is also preffered 
that you sit idle while the automation runs, only acting if it is completed 
or there is an error/problem that requires your action.

An automation may stop early only because it reached its declared cap, no
eligible candidate remains, a scored failure changes the declared eligibility
rule, or it encounters a reproducible error. Record the concrete stop reason
in `automation-end`. Never terminate a live runner based on empty or buffered
logs, and never run runner commands concurrently; wait for the active command
or its configured timeout.

### Autonomous execution and triggers

Launch an automation helper once as a foreground, blocking command, then leave
it alone until it exits. Do not tail, poll, pause, or send a progress message
after a fixed number of successful micro-trials. A qualifying micro-trial is
normal operation, **not** a trigger. Keep helpers quiet while they are making
progress: do not print one line per successful micro-trial.

The only reasons for the helper to return control are: a scored `discard`, a
recorded `error`, the declared cap, no eligible candidate, a runner nonzero
exit or timeout, or another reproducible infrastructure problem. The runner
writes `AUTOMATION_TRIGGER.json` in the active automation directory for a
scored discard/error or a reached cap. A helper must write the same JSON file
for no-candidate and runner-failure triggers, containing the reason, the last
micro-trial ID if one exists, and a concise detail. When the helper exits,
read that one trigger artifact and the final recorded result, call
`automation-end` exactly once, and then decide what to do next. Do not create
routine agent messages merely to report that the automation is working.

## What to preserve in reasoning

For each proposal, distinguish a parameterization-preserving compression from
a representational change (for example, token representation, positional
integration, deterministic/tied projections, attention organization, or
feed-forward mechanism). The source snapshots and Codex event log are research
artifacts, not scratch files.
