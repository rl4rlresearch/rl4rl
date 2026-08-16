# Improvements for the Autoresearch program

This note proposes a stronger next version of `PROGRAM.md` for the AdderBoard
Autoresearch experiment. Its aim is not merely to find a lower parameter count,
but to generate a trajectory that can support a credible research claim about
how an autonomous agent explores architectural alternatives.

## Evidence motivating the changes

The completed `sol-xhigh-overnight-20260814` run reduced the qualified model
from 6,080 to 1,831 parameters in 300 logged candidate attempts. That is a
strong optimization result, but the trajectory was concentrated:

- 273 candidate attempts were retained and 27 ended as errors.
- 217 retained candidates were scalar-pruning or fixed-zero variants.
- Only a small share of retained candidates tested substantially different
  mechanisms: position changes (16), attention changes (14), token/sequence
  changes (3), and normalization changes (2).
- Twenty-five of the 27 errors had a successful training-process exit code but
  no checkpoint suitable for official verification. These are currently
  difficult to interpret as capacity limits versus training or implementation
  failures.
- The agent used the fixed official verifier after every candidate. Repeated
  adaptive search against a single fixed evaluation seed makes the final
  score less persuasive as a generalization measurement.
- Training schedules were changed during the search, including longer
  fine-tuning budgets for more compressed candidates. This entangles
  architectural gains with additional optimization compute.

The final 1,831-parameter model did generalize to one independently generated
10,000-pair holdout with zero exact official-pair overlap and zero shared
operands: it achieved 99.84%, the same as its official score. That is good
evidence against simple memorization of the official cases, but a stronger
protocol should make this evaluation separation part of the experiment rather
than a post-hoc check.

## 1. Use explicit exploration, exploitation, and confirmation phases

The current instruction to “include representational substitutions” is too
weak: an agent can satisfy it with a few early trials, then spend most of the
budget on incremental scalar pruning.

Add language such as:

```text
Organize the run into four phases:

1. Map bottlenecks: establish coarse capacity boundaries for major components.
2. Explore representations: test materially different token, position,
   attention, feed-forward, and normalization mechanisms.
3. Exploit: spend focused effort only on mechanism families supported by prior
   evidence.
4. Confirm: independently retrain and evaluate the best frontier candidates.
```

Require a mechanism-family label in every proposal, for example:

```text
`family`: feed-forward width | token representation | position representation |
attention organization | normalization | parameter tying | scalar pruning |
training-control
```

Add a diversity guardrail:

```text
Do not make more than eight consecutive attempts from one family without
testing a materially different family. Before beginning a long exploitation
streak, state the evidence that makes that family the current priority.
```

The exact number is a design choice. It should be preregistered and held fixed
across compared agents.

## 2. Require a compact evidence review before each proposal

“Inspect `RESULTS.tsv`” is not enough. The agent should synthesize the
trajectory rather than simply continue the latest local pattern.

Add this required pre-proposal record:

```text
Before each proposal, summarize: (a) the retained frontier and its accuracy
margin, (b) the last accepted and last failed attempt in the same family,
(c) the number of attempts already spent in each family, and (d) why the
proposed change is more informative than the nearest untested alternative.
```

The runner should write this structured record to a small per-attempt JSON
file. Keeping it out of free-form agent prose makes later analysis easier.

## 3. Separate development evaluation from final evaluation

This is the most important validity change. A fixed test set is no longer a
clean final evaluation once an agent adaptively receives its result hundreds
of times.

Use three clearly separated sets:

```text
Development set: available after each attempt for diagnostic feedback.
Promotion set: accessed only when a candidate improves the development
frontier, and used sparingly.
Final holdout: never exposed during search; evaluated only after the run.
```

Use disjoint random seeds and explicitly record all generated pairs or their
reproducible seed plus hash. The final holdout should also exclude exact pairs
used by the other sets. The existing `evaluate_novel_holdout.py` script is a
starting point for this check.

This cannot be solved by `PROGRAM.md` alone: the runner must stop invoking the
official final verifier after every candidate.

## 4. Confirm frontier candidates across training and evaluation seeds

A single qualified run can be a lucky optimization outcome, especially near a
hard threshold. Distinguish a provisional frontier from a confirmed frontier.

Suggested policy:

```text
An attempt may be provisionally retained after meeting the development
threshold and reducing parameters. Before it becomes a confirmed frontier,
retrain it from at least three training seeds and evaluate each training run
on multiple held-out evaluation seeds. Report minimum, mean, and standard
deviation of accuracy.
```

For a small-budget pilot, a lighter version is reasonable: confirm only every
new global-best candidate and the final candidate.

## 5. Make training compute a controlled variable

The previous run changed fine-tuning duration as compression got harder. That
may be useful in an optimization competition, but it makes causal comparison
between architectures harder.

Add:

```text
Within a phase, keep optimizer, initialization source, data distribution,
learning-rate schedule, and step budget fixed. A training-policy change must
be a separate logged intervention; do not combine it with an architecture
change.
```

The runner should record at least:

- training seed;
- initialization/checkpoint source;
- number of optimizer steps actually executed;
- early-stopping rule and trigger;
- wall-clock training time; and
- device type.

If compute is intentionally part of the search, report a two-dimensional
frontier: parameter count and training/inference cost.

## 6. Turn errors into interpretable outcomes

An `error` with no official score is currently ambiguous. Require the agent
and runner to distinguish:

```text
infrastructure error     command, timeout, dependency, or filesystem problem
implementation error     broken model/data/submission contract
optimization failure     trained model never reached a checkpoint criterion
nonqualification         valid checkpoint reached official/development score
                         below threshold
```

Add this protocol rule:

```text
Repair a reproducible infrastructure or implementation error once without
treating it as evidence about model capacity. Do not silently retry an
optimization failure; record the learning curve and explain why a retry is
scientifically justified.
```

The runner should save the best internal validation score and checkpoint
decision reason even when no checkpoint is written.

## 7. Use an accuracy-margin policy

The formal qualification threshold can remain 99%, but candidates near it
should not automatically become the basis for extensive subsequent pruning.
The completed run retained five candidates below 99.5%, including one at
99.05%.

Suggested distinction:

```text
Provisional qualification: at least 99.0% on the development criterion.
Robust frontier: at least 99.5% development accuracy plus confirmation across
the prescribed independent runs.
```

This preserves the original objective while making it clear which small models
have meaningful safety margin.

## 8. Constrain changes to one causal factor per attempt

“One coherent candidate change” should be made more operational. A candidate
that changes model structure, parameter transfer, training duration, and
early-stopping rules at once is hard to interpret.

Add:

```text
Each attempt must declare its changed causal factor. Do not combine an
architectural modification with a training-policy modification unless the
attempt is explicitly labeled a joint intervention and is followed by a
controlled ablation separating the two effects.
```

This permits genuine representational substitutions while preserving a usable
causal record.

## 9. Preserve portability in the program itself

Run-local `PROGRAM.md` files should not contain a particular operator’s
absolute filesystem paths. From the workspace, use:

```bash
python ../run_attempt.py --run-dir .. attempt \
  --description "short factual description" \
  --proposal "mechanism hypothesis and what changed"
```

and:

```bash
python ../run_attempt.py --run-dir .. baseline
```

The configured Python executable can remain in `RUN_CONFIG.json`; the program
instructions should stay portable across machines.

## 10. Report the trajectory, not only the final model

The paper-level output should include more than the best parameter count:

- parameter count and accuracy over attempt number;
- mechanism-family allocation over time;
- accepted, nonqualified, optimization-failure, and implementation-error
  rates by family;
- time and training steps per attempt;
- number of distinct hypotheses before long exploitation streaks;
- independent confirmation results for the final frontier; and
- the exact model, reasoning setting, prompts, code revision, and evaluator
  seeds used.

These additions make it possible to compare agents as researchers rather than
only compare their final compression score.

## 11. Let the agent design bounded automated sweeps

The completed run spent 217 retained candidates on scalar-pruning or
fixed-zero variants. A long sequence of nearly identical agent turns is an
inefficient use of an LLM: the agent should choose the *search policy*, then
let deterministic code execute its repeated inner steps.

Use a two-level experimental structure:

```text
Macro-attempt: an agent decision to test a mechanism and a bounded search
policy, such as greedy QKV scalar pruning.

Micro-trial: one automatically generated candidate evaluated by that policy.
Every micro-trial remains a saved, reproducible experimental artifact.
```

For example, an agent could configure a QKV sweep as:

```text
Rank eligible QKV scalars by a declared importance measure. For each scalar in
order: fix it to zero, initialize from the current retained model, train under
the fixed development protocol, then retain or roll back. On failure, add that
index to a failure blacklist and continue. Stop at the declared micro-trial
cap, compute cap, parameter target, or consecutive-failure boundary.
```

Add this rule to `PROGRAM.md`:

```text
When testing a monotonic or repeatedly structured mechanism, prefer a bounded
automated sweep over one agent turn per variant. Before launching a sweep,
state its hypothesis, mechanism family, candidate-ordering rule, acceptance
rule, stopping rule, maximum micro-trial count, and compute budget. The sweep
must save each micro-trial and write a compact summary before the agent chooses
the next mechanism.
```

The summary should include the best retained candidate, all attempted
parameter values, acceptance/error counts, failure blacklist, training cost,
and the reason the sweep stopped. This lets the agent revise its search policy
at a meaningful boundary instead of repeatedly consuming tokens to make a
one-line variant.

This is a token-efficiency improvement, not a free-compute improvement:
training and evaluation still happen for each micro-trial. It also makes
development/final-evaluation separation more important, because a fast
automated loop can adaptively overfit a development set much more quickly.

If the paper's central question is how agents conduct research, report macro
attempts and micro-trials separately. Better still, compare a vanilla
one-candidate-per-decision condition with a condition where the agent may
author bounded sweeps; that comparison measures whether access to automation
changes exploration, token use, compute use, and the final frontier.

## Recommended priority order

1. Separate development, promotion, and final evaluation sets in the runner.
2. Add confirmation across training/evaluation seeds for frontier candidates.
3. Add phase structure, family labels, and an exploration-diversity rule to
   `PROGRAM.md`.
4. Freeze and log training-compute policy within a phase.
5. Improve error categorization and checkpoint diagnostics.
6. Add bounded automated sweeps with macro-attempt/micro-trial logging.
7. Use portable relative commands in all generated program files.
