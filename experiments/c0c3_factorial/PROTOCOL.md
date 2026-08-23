# C0–C3 protocols v1.0–v2.1

This document describes the implemented protocol. The executable contract is
`FactorialSpec`; a campaign records its canonical protocol hash, task hash,
framework hash, starting-artifact hash, and scientific-runtime hash. Authorized
in-place amendments are permitted when trajectory continuity is required. They
must preserve prior artifacts and record their exact boundary and behavior in
machine-readable provenance so later reporting can describe what actually ran.
Version 1.0 is the 100-opportunity serial paper protocol. Version 1.1 is the
separate 30-opportunity synchronized-parallel workshop pilot. Version 1.2 is
the separate 200-opportunity continuous-session Autoresearch protocol with an
intervention every tenth opportunity. Version 1.3 keeps those proposal and
conversation rules but freezes a staged, synchronized-wave execution plan:
Block 1 C0–C3 is the one-block primary analysis, while N0 and Blocks 2–3 are
dormant optional extensions. Version 1.4 keeps the same treatment and stage
scope but lets the four initially concurrent trajectories advance independently
after their simultaneous start. Results retain their protocol labels and are
not pooled as interchangeable replications. Version 1.5 changes the
subject-facing program, task boundary, and execution ownership:
agents see only an ordinary transformer-optimization job, receive a concise
condition-private result ledger, and can submit only freshly trained models
that satisfy a learned-self-attention contract; each scheduled trajectory is
controlled independently rather than by a four-run campaign process.
Version 1.6 prospectively replaces the failed 1,644-parent launch with three
predeclared blocks and a confined continuous-session runtime. It fixes resumed
process cwd ownership, isolates Codex configuration and thread identity, freezes
inference preprocessing, strengthens fresh-training/attention-dependence checks,
and caps concurrent local evaluators at three per campaign while all twelve
controllers run. Local campaigns also enter one six-slot host scheduler.
Version 2.0 is the prospective controlled OpenEvolve replacement. It removes N0,
uses the 1,644-parameter pair-token parent and 5,000-step training path, gives
each proposal a bounded ephemeral session with structured trajectory evidence,
and adds strict patch/source preflight plus optional evaluator-only Modal L4
offload. All three C0–C3 blocks are frozen primary data.
Version 1.7 is the source-only, artifact-clean continuous Autoresearch
successor. Version 2.1 applies the same subject-boundary cleanup to bounded
OpenEvolve while retaining its patch interface and evaluator controls. Both
contain twelve primary C0–C3 trajectories and no N0.

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

### Protocol 1.2–1.7 continuous-session exception

Protocols 1.2–1.7 apply only to separate Autoresearch strata. Each starts
one persisted Codex session per run at opportunity 1 and resumes that session
for all later opportunities. Before each resume, the controller reconstructs the
stable session workspace from the selected candidate, then snapshots and
evaluates only the resulting allowed files in the normal per-opportunity
workspace. The persistent transcript is therefore shared by every condition,
including C0, C2, and N0 when N0 is run. In either protocol, the C0/C2
comparison measures the extra effect of controller-provided portfolio evidence
over shared transcript memory; it is not a memory-versus-no-memory estimate.
N0 has no controller search state but is not transcript-free.

Protocol 1.5 additionally renders a subject-neutral program profile. The agent
is not shown task benchmark names, protocol/study/condition labels, treatment
terminology, internal candidate IDs, or scheduled-intervention labels. Its
Codex working directory and seed environment variable are also neutralized.
Internal metadata remains complete in the campaign outside the subject
workspace.

Every non-N0 subject-neutral opportunity receives the same bounded, condition-private
summary of up to twelve preceding outcomes from its own trajectory. This is a
common online evidence channel, not a fifth factorial factor. Portfolio cells
still differ only by access to multiple live qualified source branches.

### Protocols 2.0–2.1 bounded OpenEvolve exception

Protocol 2.0 uses a fresh ephemeral Codex call for every proposal. It provides
structured continuity common to all four cells: the selected source, all
retained source branches permitted by the search-state factor, the last twelve
outcomes, and a bounded ledger of subject-defined mechanism names and results.
It never resumes raw conversation history.

The subject-neutral v2 prompt and opaque read-only Codex cwd expose no benchmark,
study, treatment, condition, or C0–C3 labels. OpenEvolve's prompt sampler and
SEARCH/REPLACE representation remain, but native database selection and
retention remain disabled because those are the randomized C0–C3 factors.

Protocol 2.1 retains the ephemeral proposal boundary but removes resource and
horizon fields, prompt-region markers, empty design slots, selection counts,
private evaluator fields, raw runner errors, fake source paths, and redundant
mechanism/formatting sections. It supplies source-only starting material and
does not expose a checkpoint or run-derived seed. See
`ARTIFACT_CLEAN_PROTOCOLS.md` for the complete subject boundary.

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

The paper-v1 schedule is opportunities `20, 40, 60, 80`; the ephemeral
parallel-pilot schedule is `10, 20`; and the continuous Autoresearch schedule
is every tenth opportunity from `10` through `200`. C0 and C2 never receive the
transition text. The continuous Autoresearch protocol uses its own
evidence-conditioned transition template; it does not alter the prompt used by
the other protocols or by OpenEvolve. Compliance is not inferred from the
instruction label: blinded Layer B reviewers inspect parent-to-candidate
changes and decide whether a coherent mechanism change occurred.

## 7. N0 no-search baseline

N0 is a separate descriptive baseline, one per block. Each proposal starts from
the frozen seed, receives no prior proposal, metric, or trajectory feedback, and
cannot change later N0 prompts. At the end of search, the best Layer-A-valid N0
candidate may be selected offline for Layer C. This post-search selection does
not make proposal generation adaptive. N0 is reported separately and excluded
from all factorial contrasts.

Protocols 1.7, 2.0, and 2.1 contain no N0 assignment, dormant stage, output row,
or compute budget. Their campaign manifests must record
`include_no_search=false`; creation and validation fail closed if an N0 run is
requested or present.

## 8. Randomization, blocking, and seeds

For each task × framework × block, a deterministic SHA-256-derived block seed
shuffles C0–C3 once. The four cells share the same block seed. N0 is assigned
the fifth frozen order.

Protocol 1.0 freezes
`blocked_round_robin_one_opportunity_v1`: select the least-advanced run, then
block and frozen within-block order. Thus one opportunity is accrued per run per
round instead of completing one entire condition before another.

Protocol 1.1 freezes `blocked_parallel_condition_rounds_v1`. At each wave:

1. find the minimum `proposals_used` among every non-completed campaign run;
2. select the earliest block containing a run at that minimum;
3. launch every C0–C3 cell in that block at the minimum concurrently, behind
   one process-local start barrier;
4. wait for all launched factorial cells to finish; and
5. only then run an eligible N0 opportunity serially.

The campaign-wide minimum is a hard round barrier: no non-completed run advances
to a later wave while another non-completed run remains behind. Budget-completed
runs leave the eligible set under the ordinary stopping rule. One campaign
writer lock covers selection and execution. `parallel-rounds.jsonl` records the
participants and completion of each wave. Parallelism is currently supported
only for local task backends; Modal remains protocol-1.0 serialized execution.

Protocol 1.3 freezes
`staged_parallel_block_trajectories_v1`. Its primary stage is Block 1 C0–C3
only. The four conditions advance together at their within-stage minimum and
N0 does not run. That primary stage must complete before the runner permits an
optional extension. Block 1 N0, Block 2 C0–C3/N0, and Block 3 C0–C3/N0 are
pre-created with the same protocol, task, framework, calibration, runtime, and
deterministic block seeds, but advance only through an explicit staged command.
Every wave records its stage in `parallel-rounds.jsonl`.

Protocol 1.4 freezes
`staged_independent_parallel_trajectories_v1`. It has the same one-block
primary scope, run seeds, C0–C3 treatments, continuous-session rule, budgets,
and dormant optional extensions as protocol 1.3, but is an independently
advancing execution design: C0–C3 launch together behind one initial local
barrier and each worker immediately begins its next opportunity when its own
previous evaluator completes. It does not wait for a peer at opportunities
1–200. One campaign writer still owns all four workers; manually launching four
run commands remains invalid. `independent-trajectories.jsonl` records the
initial participant set, starting opportunity for each run, each completed
trajectory, and batch completion/failure. A transport failure or unexpected
runner exception prevents new opportunities from starting after in-flight work
finishes; completed records remain charged and ordinary recovery rules apply.

This changes the execution/timing instrument, not the C0–C3 treatment
boundary. Because a trajectory can get ahead of another, protocol 1.4 process
metrics must report per-run wall time and overlap rather than treating a shared
round as the unit of execution. Its factorial outcomes remain run-level and
descriptive for its one-block primary stage.

Protocol 1.5 freezes
`staged_individually_controlled_trajectories_v1`. Its new task adapter packages
only `src/model.py`, `src/data.py`,
`src/train.py`, the immutable seed checkpoint, and a protected neutral decoder.
It excludes the seed repository's reports and handoff notes, which contain
prior search suggestions and parameter targets. Before accuracy verification,
the evaluator rejects direct arithmetic/transducer source patterns, zero-scalar
saved models, models without positive training provenance, and models whose
forward pass does not exercise a learned self-attention module. These checks
are identical for C0–C3 and N0.

Each scheduled trajectory is started and resumed by its own explicit controller
command. A controller holds only that run's exclusive lock; it neither owns nor
blocks its C0–C3 peers. The only campaign-wide lock is a short stage-gate check,
which prevents N0 from beginning before the required factorial trajectories
finish, and prevents collection after Layer B/C is sealed. Predeclared
factorial C0–C3 trajectories in different blocks may start concurrently; blocks
do not share a controller or wait for one another.

A pause is cooperative and occurs only between completed opportunities. The
operator requests it from a separate terminal; the active Codex/evaluator call
finishes and remains charged, then the run records `trajectory_paused` before
another proposal can start. Resume uses the same persistent Codex session and
next opportunity. Interrupting a controller process is not a pause: use the
ordinary recovery rule if it leaves an active opportunity.

The scheduled run ID, initial start time, every pause request/acknowledgement,
resume, stop reason, and completion are append-only provenance in both the run
directory and `trajectory-lifecycle.jsonl`. Individual launch timing is an
operator-controlled execution covariate, not a C0–C3 treatment. Start all four
primary trajectories under one predeclared operational plan, do not alter a
peer's launch/pause plan based on another run's online outcomes, and report
per-run lifecycle timing and overlap.

The Block 1 factorial is the prespecified primary analysis. Any additional
block must be declared before its first opportunity and is a compatible
replication under the same instrument; predeclared blocks may run concurrently.

Protocol 1.6 freezes
`staged_confined_individually_controlled_trajectories_v1`. All three blocks are
declared before launch and all twelve C0–C3 controllers may advance
independently. Each run owns a unique persisted Codex thread recorded in an
atomic campaign registry. Initial and resumed invocations execute with the
opaque workspace as their real operating-system cwd; user config, project/user
rules, online access, and writable temporary roots outside that workspace are
disabled. A mismatched or reused thread ID invalidates the proposal.

The v1.6 task freezes `src/data.py` as protected inference preprocessing; only
`src/model.py` and `src/train.py` are candidate files. Candidate evaluation
removes every supplied checkpoint before training, rejects source mutation
during training, requires a positive-step learned checkpoint, and checks that
exact addition accuracy collapses when learned attention is ablated. At most
three v1.6 evaluators hold campaign-local file-lock slots concurrently. Every
local protocol-1.6/1.7/2.0/2.1 evaluator must also hold one of six host-wide
file-lock slots shared across campaigns. Both leases release automatically on
process exit. Waiting for either slot occurs outside evaluator timeout and
evaluator-budget accounting; Codex proposal generation and trajectory
ownership remain fully parallel.
If a block was activated using primary results, it is adaptively collected and
must not be presented as if the original primary sample size had always
included it. N0 remains descriptive and outside the factorial contrasts.

Protocol 1.6 uses 500,000,000 reported tokens as a common subject-visible phase
threshold rather than a hard stopping budget. Before a run reaches the
threshold, prompts expose the ordinary decreasing token remainder. Crossing the
threshold ends that controller invocation. On the first resumed opportunity,
the prompt omits token accounting and says only that the run may continue past
the previously stated token budget. That notice is recorded once. Every later
prompt omits all token-budget language. The rule is identical for all twelve
C0–C3 trajectories; proposal, evaluation, and evaluator-time budgets remain
hard limits.

Protocol 1.7 uses the same independently controlled, confined continuous
execution geometry but freezes all three C0–C3 blocks as primary scope and
removes N0. The subject receives the complete task contract once, followed by
incremental current-design and newly available result messages. Prompt-region
markers, opportunity/resource accounting, empty population slots, selection
counts, nonpublic metrics, internal runner errors, and repeated history are not
subject-visible. Treatment skeletons remain controller-auditable through
prompt-manifest hashes.

The v1.7 task-support tree contains no starting checkpoint. The subject sees
source plus verified public results; calibration and candidate evaluation train
in separate workspaces from fresh initialization. Its Codex process receives no
block/run seed or workspace identity file. The controller provides a local Git
baseline and workspace-confined caches to keep ordinary source-inspection
commands useful without adding study metadata. Token usage is recorded but
does not stop the 200-opportunity trajectory and is never mentioned in a
subject prompt.

Protocol 2.0 freezes
`confined_individually_controlled_c0c3_only_trajectories_v2`. All twelve C0–C3
trajectories across three blocks are declared before launch and are controlled,
paused, resumed, and recovered independently. There is no block or opportunity
barrier. At most three evaluators from this campaign may train concurrently,
and every local evaluator also enters the shared six-slot host scheduler;
proposal generation remains independently parallel. A durable supervisor may relaunch only a
trajectory whose writer is confirmed absent, using the charged recovery rule.

Only `src/model.py` and `src/train.py` are editable. Every OpenEvolve patch block
must be well formed, nonempty, and match exactly once after earlier blocks are
applied. Syntax and recognized direct-solver patterns are checked before an
evaluator call. Candidate training starts from an empty checkpoint directory,
may run for at most 1,800 seconds, and must produce both a valid best checkpoint
and a positive-step last checkpoint. Training-time source hashes, learned-state
checks, exercised self-attention, attention ablation, and disjoint Layer C are
mandatory and condition-common.

Protocol 2.1 retains v2.0's execution, strict patching, source preflight,
fresh-training checks, and evaluator limits. It uses the same source-only task
boundary as v1.7 and removes subject-visible controller state. Public design
evidence and reference source are each rendered once; recent outcomes carry
mechanism information without a second mechanism ledger; the patch/metadata
contract appears once. Its internal token ceiling remains a safety stop but is
not disclosed to the subject.

The shared local scheduler is an operational machine-load control, not a
factorial treatment. It preserves every campaign's three-slot ceiling while
enforcing six local evaluations across all participating campaigns combined.
It does not schedule Codex calls, alter proposal order or parent selection,
charge queue time, or change evaluator seeds, commands, timeouts, and budgets.
Evaluator-only Modal calls retain their campaign remote-call limit but do not
consume a local host slot.

`C0C3_RUN_SEED` and `PYTHONHASHSEED` are supplied to evaluator and legacy
non-neutral subprocesses. Task code may use `C0C3_RUN_SEED`; task-specific fixed
seeds still take precedence where declared. The Codex provider does not expose
a generation seed through this runner, so model sampling is not deterministic.
Blocking, identical settings, and replication mitigate that source of variance;
they do not eliminate it.

The protocol-1.5/1.6/2.0 subject-facing Codex process instead receives the same
value under `OPTIMIZATION_RUN_SEED`; `C0C3_RUN_SEED` is removed from that
process. Protocols 1.7 and 2.1 expose neither seed name nor a run-derived
`PYTHONHASHSEED` to the subject. Evaluators retain the internal seed for
controlled task execution.

Use `run-next` or `run-campaign` for protocol 1.0. Use `run-parallel-next` or
`run-parallel-campaign` for protocol 1.1. Direct `run-one --run-id` is diagnostic
and can bypass randomized order.
Use `run-staged-next` or `run-staged-campaign` for protocol 1.3. Use
`run-staged-independent-campaign` for protocol 1.4. Use the
per-run `start-staged-trajectory`, `pause-staged-trajectory`, and
`resume-staged-trajectory` commands for protocols 1.5–1.7 and 2.0–2.1; the other
orchestrators reject their execution-rule identifiers.

## 9. Controls held common within a campaign

- Codex model, reasoning effort, sandbox, approval policy, and conversation mode.
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

The workshop parallel pilot freezes, per run, 30 opportunities/evaluator calls,
30,000,000 total reported tokens, 108,000 aggregate evaluator seconds, and the
same 3,600-second evaluator timeout. It uses three blocks and therefore has
three independent trajectories per factorial cell plus three separate N0 runs.

Protocols 1.2–1.5 freeze 200 opportunities/evaluator calls, 500,000,000
total reported tokens, 720,000 aggregate evaluator seconds, and the same
3,600-second evaluator timeout per run. Protocols 1.3–1.5 primary stages
contain four runs and therefore schedule 800 proposals and 40
assumption-changing interventions. Dormant extension runs consume no budget
until explicitly activated.

Protocol 1.6 freezes 200 opportunities/evaluator calls and 720,000 aggregate
evaluator seconds per run. Its 500,000,000-token value is the two-phase prompt
threshold described above, not a total-token stopping ceiling.

Protocol 1.7 freezes 200 opportunities/evaluator calls and 720,000 aggregate
evaluator seconds per run. Token use is accounted but has no stopping or prompt
role. Its three C0–C3 blocks schedule 2,400 proposals and contain no N0.

Protocols 2.0 and 2.1 freeze 200 opportunities/evaluator calls, 100,000,000 total
reported tokens, 360,000 aggregate evaluator seconds, and 1,800 seconds per
evaluator invocation per run. It schedules 2,400 proposals across twelve runs
and twenty assumption-changing opportunities per C1/C3 trajectory. There is no
N0 budget. The lower token ceiling reflects bounded ephemeral prompts rather
than an intended difference among cells.

A new opportunity cannot start after any applicable hard budget reaches zero.
One already-started Codex or evaluator call may overshoot a hard ceiling; its
actual usage is logged and the run then completes. Protocol 1.6's token
threshold instead returns once and resumes under its post-threshold prompt
phase; protocol 1.7 has no token stop or phase. There is no performance-based
early stop.

## 11. Failures and recovery

Provider, formatting, protected-file, duplicate, no-change, evaluator timeout,
execution, and nonqualification failures are recorded and never retained. A
proposal opportunity is always consumed after it starts. An evaluator call is
consumed only if evaluation started.

An infrastructure interruption can leave `state.active` populated. The only
valid continuation is `recover-active`, which logs the reason, recovers any
available Codex usage, records a zero-evaluator infrastructure failure, consumes
the opportunity, and preserves all artifacts. Never delete or retry it.

Under protocol 1.1, first recover every active opportunity explicitly. A host
interruption can also occur before every intended peer starts. The frozen
campaign-wide-minimum rule then launches only the still-lagging C0–C3 cells;
`recovery_subset=true` in `parallel-rounds.jsonl` discloses that partial-round
completion. Previously completed cells are never repeated and N0 still waits
until the selected factorial subset finishes.

Under protocol 1.3, recover every active opportunity and resume the same
explicit block/stage. Only lagging peers in that stage are selected. Under
protocol 1.4, recover every active opportunity and resume the same independent
stage launcher; every unfinished selected trajectory resumes from its own next
opportunity, without creating a new wave barrier. Under protocol 1.5, recover
only the affected run, then use `resume-staged-trajectory` for that run ID.
Protocols 1.6 and 1.7 use the same charged recovery rule; their durable supervisors
automatically performs that explicit recovery after an unexpected controller
exit and restart only the affected run. Protocol 1.6 does not advance N0 or an
undeclared extension; protocol 1.7 has no N0 or optional block.
Protocols 2.0 and 2.1 use the same charged recovery and per-run supervisor
behavior.
Malformed patches, source failures, nonqualification, and model-contract
failures are never retried. A remote evaluator transport failure is recorded
separately as infrastructure; it still consumes the started proposal and
evaluator call and is not silently replayed.

## 12. Feedback layers

### Layer A — online

Older protocols may expose declared public metrics, retained source, selection
counts, hypotheses, parent selection, and remaining budgets. Protocols 1.7 and
2.1 expose only public metrics, available source/design evidence, hypotheses,
and sanitized subject-level outcome explanations. They never expose selection
counts, internal fitness, budgets, horizons, or raw infrastructure fields.

### Layer B — sealed mechanism review

Layer B cannot be exported until every run in the frozen analysis scope is
completed. For protocols 1.3–1.5 that scope is the four Block 1 primary runs plus
every explicitly activated and completed extension stage; dormant extensions
are excluded. The operator must activate any optional extension before Layer
B/C is created, after which the runner forbids more collection.
For protocol 1.6 the frozen factorial scope is all twelve C0–C3 runs across its
three prospectively declared blocks; dormant N0 assignments remain excluded.
For protocols 1.7, 2.0, and 2.1 the frozen scope is all twelve C0–C3 runs and no
dormant N0 assignment exists.
Every valid proposal becomes an opaque packet containing parent and candidate
source plus the stated hypothesis/edit. Condition, run, opportunity, and Layer
A scores are hidden. Independent reviewers decide whether the delta is a
coherent testable mechanism change and assign a stable cross-packet cluster
label. Repetitions of one mechanism count once per run.

Use at least two independent reviewers, report raw agreement, adjudicate
disagreements without treatment labels, and score only the frozen adjudicated
file. See `PAPER_NOTES.md` for the rubric.

### Layer C — sealed final evaluation

Layer C also waits for all runs in the frozen analysis scope. Factorial cells
evaluate the online incumbent. N0 evaluates its post-search best Layer-A-valid
independent proposal when N0 is included in that scope. AdderBoard uses a
disjoint verifier seed. Official Karpathy Autoresearch currently repeats the
pinned validation procedure, so it is a replication check, not an unseen
holdout; the paper must use that terminology.

## 13. Confirmatory analysis

Analyze every task × framework × protocol stratum separately first. For each
stratum:

1. Publish the four cell means and all run-level counts.
2. Publish the three prespecified contrasts and within-block contrasts generated
   by `analysis.py`.
3. Show N0 separately when the protocol includes it; protocols 1.7, 2.0, and
   2.1 have no N0.
4. Report invalid-proposal rate, qualification rate, tokens, evaluator calls,
   evaluator time, and termination reason as process/efficiency outcomes.
5. Do not treat proposals as independent samples or inflate `n` with packets.

With three blocks per cell, uncertainty is necessarily wide. Emphasize effect
sizes, raw blocks, and compatibility intervals rather than binary significance.
Any pooled synthesis across tasks/frameworks should be hierarchical or a
clearly labeled descriptive meta-analysis, not a replacement for stratum-level
results.

Protocols 1.3–1.5 primary stages have one trajectory per cell. Their cell values and
three factorial contrasts are descriptive: there is no between-block sampling
variance estimate and proposals cannot be substituted as replicates. If later
blocks are activated, publish the original Block 1 result unchanged, then show
the extension blocks and a transparently labeled combined sensitivity analysis.

## 14. Exclusions and deviations

No completed run or valid proposal is excluded based on performance. Incomplete
runs are not imputed into the confirmatory estimate. If external failure makes a
campaign incomplete, report it and either finish under the frozen recovery rule
or label the campaign non-confirmatory.

Changes after the first run—including model alias behavior, prompts, `K`,
schedule, selection, retention, evaluator, public metrics, budgets, timeout,
task source, dependency environment, or runtime code—may be applied as
operator-authorized in-place amendments. Preserve the pre-change artifacts,
record the affected run IDs and first affected opportunity, and make the
post-change behavior explicit. A new protocol version, campaign, calibration,
or analysis stratum is optional rather than automatic; choose those boundaries
according to the scientific question and disclose the executed history in any
analysis that uses the affected data.
