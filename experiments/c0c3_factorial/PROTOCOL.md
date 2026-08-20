# C0–C3 protocol v1.0

This document is the human-readable preregistration for the implementation in
this directory. The executable contract is `FactorialSpec`; a campaign records
its canonical protocol hash, task hash, framework hash, starting-artifact hash,
and scientific-runtime hash. Disagreement between this document and executable
state is a protocol deviation that must be disclosed, not silently repaired.

## 1. Research question and unit of analysis

The experiment asks whether autonomous ML-research agents produce more
distinct, task-valid mechanism families when they have:

1. portfolio memory rather than a single incumbent;
2. fixed assumption-changing checkpoints rather than only ordinary proposals;
3. both interventions together.

The experimental unit is one complete trajectory (“run”) within a fixed task ×
framework × block stratum. A proposal is not an independent replicate. The
primary outcome is the number of distinct Layer-B-qualified mechanism clusters
among all Layer-A-valid proposals produced by that run, including valid
proposals that were not retained online.

## 2. Factorial cells

| Condition | Search state | Proposal policy |
|---|---|---|
| C0 | Single incumbent | Ordinary at every opportunity |
| C1 | Single incumbent | Assumption-changing at frozen checkpoints |
| C2 | Portfolio memory | Ordinary at every opportunity |
| C3 | Portfolio memory | Assumption-changing at the same checkpoints as C1 |

The memory main effect is the average of C2/C3 minus the average of C0/C1. The
proposal-policy main effect is the average of C1/C3 minus the average of C0/C2.
The interaction is `(C3 - C2) - (C1 - C0)`.

## 3. One proposal opportunity

Opportunity indices are one-based. Each opportunity performs exactly this
sequence:

1. Atomically mark one opportunity active and charge no resources yet.
2. Select one parent using the condition’s frozen rule.
3. Materialize the parent as the editable workspace and each visible candidate
   as a read-only evidence workspace.
4. Render the common prompt and the two treatment regions.
5. Invoke one fresh, ephemeral Codex CLI process.
6. Snapshot the allowed editable files by content hash.
7. Reject without evaluation on provider failure, no edit, duplicate candidate,
   malformed framework output, or protected-file modification.
8. Otherwise invoke the evaluator once at most.
9. Apply the frozen retention rule, append the complete event, atomically save
   state, and advance the opportunity index.

No conversation is resumed between opportunities. Cross-opportunity memory is
exactly the code, metrics, selection counts, and hypotheses exposed by the
controller.

## 4. Single-incumbent state (C0/C1)

Only the incumbent is visible and selected. A valid child replaces it if and
only if normalized fitness is strictly greater than the incumbent’s. Ties,
worse results, nonqualification, and failures do not alter visible state.
Rejected candidates remain in immutable logs but are never shown online.

## 5. Portfolio state (C2/C3)

The paper protocol freezes `K=4` and these versioned rules:

- Retention:
  `fill_open_slots_then_replace_selected_lineage_on_strict_improvement_v1`.
- Parent selection:
  `fill_from_seed_then_least_selected_lineage_then_best_then_oldest_then_id_v1`.

### Fill phase

The frozen seed remains the editable parent until all `K` slots are filled.
Every Layer-A-valid child fills one open slot, even when its fitness is below the
seed. This creates comparable initial branches instead of letting a newly added
zero-count child immediately turn filling into one deep chain. Invalid proposals
consume opportunities but do not fill slots.

### Full portfolio

Select the retained lineage with:

1. lowest cumulative selection count;
2. highest normalized fitness;
3. earliest retained order;
4. lexicographically smallest candidate ID.

A valid child replaces only its selected parent and only on strict improvement.
The child inherits the parent lineage’s cumulative selection count, including
the current selection. This prevents successful replacement from resetting to
zero and monopolizing the next opportunity. The globally best retained member
is tracked as the incumbent for final selection but is not always the parent.

All retained slots are visible; only the selected parent is editable.

## 6. Proposal-policy treatment

Outside frozen checkpoints, every cell receives the same ordinary instruction:
make the normal best next coherent experiment from visible Layer A evidence.

At checkpoints, C1 and C3 instead receive the exact same instruction to identify
and challenge a core assumption and test a meaningfully different architecture
family. Width/depth changes, scalar/hyperparameter tuning, deletion alone, or a
renamed instance of the same computation do not satisfy that instruction.

The paper schedule is opportunities `20, 40, 60, 80`. C0 and C2 never receive
the transition text. Compliance is not inferred from the instruction label:
blinded Layer B reviewers inspect parent-to-candidate changes and decide whether
a coherent mechanism change occurred.

## 7. N0 no-search baseline

N0 is a separate descriptive baseline, one per block. Each proposal starts from
the frozen seed, receives no prior proposal, metric, or trajectory feedback, and
cannot change later N0 prompts. At the end of search, the best Layer-A-valid N0
candidate may be selected offline for Layer C. This post-search selection does
not make proposal generation adaptive. N0 is reported separately and excluded
from all factorial contrasts.

## 8. Randomization, blocking, and seeds

For each task × framework × block, a deterministic SHA-256-derived block seed
shuffles C0–C3 once. The four cells share the same block seed. N0 follows as the
fifth run. Scientific execution uses blocked round-robin order: the least-
advanced run is selected first, then block and frozen within-block order. Thus
one opportunity is accrued per run per round instead of completing one entire
condition before another.

`C0C3_RUN_SEED` and `PYTHONHASHSEED` are supplied to Codex and evaluator
subprocesses. Task code may use `C0C3_RUN_SEED`; task-specific fixed seeds still
take precedence where declared. The Codex provider does not expose a generation
seed through this runner, so model sampling is not deterministic. Blocking,
identical settings, and replication mitigate that source of variance; they do
not eliminate it.

Use `run-next` or `run-campaign`. Direct `run-one --run-id` is diagnostic and can
bypass randomized order.

## 9. Controls held common within a campaign

- Codex model, reasoning effort, sandbox, approval policy, and ephemeral mode.
- Starting task-support tree and seed candidate, verified by content hash.
- Editable paths, evaluator command, objective, qualification, feedback fields,
  timeout, and target backend.
- Common prompt template and task/framework contract.
- Available filesystem/tools for the chosen framework adapter.
- Proposal, evaluator-call, token, evaluator-time, and per-call timeout budgets.
- One-evaluator-call maximum and failure handling.
- Layer A field definitions.
- Scientific runtime code, verified before every opportunity.

Frameworks and tasks are separate experimental strata. OpenEvolve and direct
Autoresearch have different edit interfaces by definition; all four conditions
within one framework use the identical interface.

## 10. Budget and stopping

Paper v1 freezes, per run:

- 100 proposal opportunities;
- at most 100 evaluator calls;
- 100,000,000 uncached-plus-cached-input-inclusive input tokens plus output
  tokens as reported by Codex (`input_tokens + output_tokens` total; cached input
  is a reported subset and is not double-counted);
- 360,000 aggregate evaluator seconds;
- 3,600 seconds per evaluator invocation.

A new opportunity cannot start after any remaining budget reaches zero. One
already-started Codex or evaluator call may overshoot a ceiling; its actual usage
is logged and the run then completes. There is no performance-based early stop.

## 11. Failures and recovery

Provider, formatting, protected-file, duplicate, no-change, evaluator timeout,
execution, and nonqualification failures are recorded and never retained. A
proposal opportunity is always consumed after it starts. An evaluator call is
consumed only if evaluation started.

An infrastructure interruption can leave `state.active` populated. The only
valid continuation is `recover-active`, which logs the reason, recovers any
available Codex usage, records a zero-evaluator infrastructure failure, consumes
the opportunity, and preserves all artifacts. Never delete or retry it.

## 12. Feedback layers

### Layer A — online

Only declared public metrics, retained source, selection counts, hypotheses,
parent selection, and remaining budgets are visible during search. For a
minimization task the internal sign-normalized fitness may be shown alongside
the human-readable metric.

### Layer B — sealed mechanism review

Layer B cannot be exported until every campaign run is completed. Every valid
proposal becomes an opaque packet containing parent and candidate source plus
the stated hypothesis/edit. Condition, run, opportunity, and Layer A scores are
hidden. Independent reviewers decide whether the delta is a coherent testable
mechanism change and assign a stable cross-packet cluster label. Repetitions of
one mechanism count once per run.

Use at least two independent reviewers, report raw agreement, adjudicate
disagreements without treatment labels, and score only the frozen adjudicated
file. See `PAPER_NOTES.md` for the rubric.

### Layer C — sealed final evaluation

Layer C also waits for all runs. Factorial cells evaluate the online incumbent.
N0 evaluates its post-search best Layer-A-valid independent proposal. AdderBoard
uses a disjoint verifier seed. Official Karpathy Autoresearch currently repeats
the pinned validation procedure, so it is a replication check, not an unseen
holdout; the paper must use that terminology.

## 13. Confirmatory analysis

Analyze every task × framework stratum separately first. For each stratum:

1. Publish the four cell means and all run-level counts.
2. Publish the three prespecified contrasts and within-block contrasts generated
   by `analysis.py`.
3. Show N0 separately.
4. Report invalid-proposal rate, qualification rate, tokens, evaluator calls,
   evaluator time, and termination reason as process/efficiency outcomes.
5. Do not treat proposals as independent samples or inflate `n` with packets.

With three blocks per cell, uncertainty is necessarily wide. Emphasize effect
sizes, raw blocks, and compatibility intervals rather than binary significance.
Any pooled synthesis across tasks/frameworks should be hierarchical or a
clearly labeled descriptive meta-analysis, not a replacement for stratum-level
results.

## 14. Exclusions and deviations

No completed run or valid proposal is excluded based on performance. Incomplete
runs are not imputed into the confirmatory estimate. If external failure makes a
campaign incomplete, report it and either finish under the frozen recovery rule
or label the campaign non-confirmatory.

Any change after the first paper run—including model alias behavior, prompts,
`K`, schedule, selection, retention, evaluator, public metrics, timeout, task
source, dependency environment, or runtime code—creates a new protocol version.
Do not merge those runs into paper v1 without an explicit sensitivity analysis
and deviation record.
