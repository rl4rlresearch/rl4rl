# Candidate improvements for the next Autoresearch and OpenEvolve protocols

**Status:** design backlog only. Nothing in this document is implemented,
enabled, frozen, or authorized for an existing campaign. It does not change the
meaning or execution of Autoresearch v1.7, OpenEvolve v2.1, or any active run.

**Evidence cutoff:** 2026-08-24. The live campaigns were incomplete when the
diagnostic counts below were taken, so those counts are observations for
designing future protocols, not final treatment-effect estimates.

## 1. How to use this backlog

The entries are deliberately broader than one sensible version increment. Do
not implement all of them together. A useful next protocol should choose a
small coherent subset, freeze it before launch, and create fresh prospective
campaigns. Changes that answer different scientific questions should become
separate protocol or framework strata rather than a single complicated system.

Priorities used below:

- **P0 — launch gate:** fix before spending heavily on a successor campaign.
- **P1 — strong improvement:** likely to materially improve validity,
  interpretability, or efficiency.
- **P2 — useful extension:** valuable after the core design is stable.
- **Research variant:** changes the estimand enough that it should be a separate
  experiment, not silently folded into the default successor.

Every implemented item should be checked against four rules:

1. Keep it condition-common unless it is the explicitly randomized treatment.
2. Freeze its behavior and provenance before the first affected trajectory.
3. Preserve started opportunities and append-only records during recovery.
4. Analyze task × framework × protocol strata separately before synthesis.

## 2. What the current runs already suggest

These diagnostics are the main reasons some items are ranked above others.

### 2.1 Continuous Autoresearch has severe transcript-cost growth

| Campaign | Completed proposals in snapshot | Accounted tokens | Median tokens/proposal early | Median tokens/proposal later |
|---|---:|---:|---:|---:|
| Addition Autoresearch v1.7 | 193 | 723.5M | 0.99M at opportunities 1–10 | 12.57M at 31–40 |
| nanoGPT Autoresearch v1.7 | 139 | 124.6M | 0.77M at 1–10 | 2.26M at 11–20, with only three observations |
| Fashion-MNIST Autoresearch v1.7 | 934 | 9.70B | 0.76M at 1–10 | 29.78M at 71–80 |
| Addition OpenEvolve v2.1 | 1,025 | 33.4M | 22.3k at 1–10 | 47.8k at 61–70 |
| nanoGPT OpenEvolve v2.1 | 659 | 24.8M | 28.9k at 1–10 | 48.0k at 51–60 |
| Fashion-MNIST OpenEvolve v2.1 | about 2,348 | 54.1M | 21.2k at 1–10 | roughly 22–24k through 150 |

Most Autoresearch input is cached, but it still consumes rate-limit capacity
and has a real price/capacity cost. Fashion-MNIST Autoresearch recorded 9.36B
cached-input tokens and 58.96M output tokens in this snapshot. The roughly
linear-to-superlinear per-opportunity growth is consistent with repeatedly
resuming an ever-growing transcript, not with the size of the short v1.7
continuation prompt.

This is both an efficiency problem and a construct problem: C0/C1 still retain
raw conversational memory of rejected ideas and old source states, so the
Autoresearch comparison is “extra controller portfolio evidence versus a
single continuous transcript,” not “memory versus no memory.”

### 2.2 The assumption-changing intervention is useful but risky

| Campaign | Ordinary valid | Assumption-changing valid | Ordinary retained | Assumption-changing retained |
|---|---:|---:|---:|---:|
| Addition Autoresearch v1.7 | 102/185 (55.1%) | 4/8 (50.0%) | 102/185 | 4/8 |
| Fashion-MNIST Autoresearch v1.7 | 688/890 (77.3%) | 26/44 (59.1%) | 318/890 (35.7%) | 7/44 (15.9%) |
| Addition OpenEvolve v2.1 | 513/975 (52.6%) | 21/50 (42.0%) | 512/975 (52.5%) | 20/50 (40.0%) |
| nanoGPT OpenEvolve v2.1 | 595/630 (94.4%) | 27/29 (93.1%) | 241/630 (38.3%) | 8/29 (27.6%) |
| Fashion-MNIST OpenEvolve v2.1 | 1,223/2,237 (54.7%) | 32/109 (29.4%) | 653/2,237 (29.2%) | 10/109 (9.2%) |

The prompt can produce genuine mechanism changes. Addition OpenEvolve
transition proposals have found qualified low-rank representations, harmonic
position encodings, and shared-value or multi-query attention designs. It also
causes large destructive jumps, and on Fashion-MNIST it disproportionately
proposes computationally expensive mechanisms that hit the wall-time guard.

The right conclusion is not simply “make the intervention stronger” or “remove
it.” The next protocol should preserve genuine assumption changes while making
feasibility, manipulation checks, and timing part of the design.

### 2.3 Fashion-MNIST timeout behavior is a major validity threat

- Fashion-MNIST Autoresearch recorded 206 timeouts among 934 proposals.
- Fashion-MNIST OpenEvolve recorded 899 timeouts among roughly 2,348 proposals.
- Addition OpenEvolve recorded only 30 timeouts among 1,025 proposals.
- The current Mac can run many MPS evaluators simultaneously, but concurrent
  processes contend for the same GPU, memory bandwidth, CPU, and thermal
  envelope. A twelve-slot host cap prevents unbounded process creation; it does
  not make twelve MPS trainings performance-isolated.

Because assumption-changing proposals often add attention, residual paths,
ensembles, covariance features, or other compute, contention can turn a real
proposal-policy effect into a timeout effect. This must be addressed before
using another local-MPS campaign as clean causal evidence.

### 2.4 OpenEvolve patch formatting is no longer a leading problem

The large Fashion-MNIST OpenEvolve snapshot contained only one ambiguous patch
and one unmatched patch. Spending substantial protocol complexity on a new
patch format would have little expected return. Execution, timeout, candidate
validity, evidence memory, and scientific diversity matter more.

### 2.5 Free-form mechanism labels are provenance, not clusters

OpenEvolve produced nearly one unique free-form mechanism string per proposal
(for example, 1,824 distinct non-placeholder labels in the Fashion-MNIST
snapshot). These strings are useful descriptions but do not deduplicate
mechanistic equivalence. Autoresearch v1.7 records no equivalent structured
mechanism field. Neither fact breaks blinded Layer B, but both limit online
repeat detection and process analysis.

### 2.6 Portfolio context has an inherent cost

OpenEvolve portfolio cells consumed materially more tokens per proposal than
single-incumbent cells because they receive additional source branches. The
difference was roughly 26k versus 48k on nanoGPT and roughly 20–21k versus
24–26k on Fashion-MNIST. That cost can be considered part of a real portfolio
system, but it must not be mistaken for a free treatment. A future design must
choose whether it estimates the effect of the whole portfolio system or the
effect of branch diversity under approximately equal context size.

## 3. P0 shared launch gates

### 3.1 Isolate evaluator performance from condition and host load

1. **Calibrate safe concurrency empirically.** Benchmark one, two, three, and
   more simultaneous evaluations using representative fast and slow candidates.
   Freeze the highest setting whose runtime and result distribution remain
   equivalent to isolated evaluation.
2. **Use one MPS evaluator at a time unless calibration supports more.** The Mac
   has one integrated GPU; process slots are not independent accelerators.
3. **Create task-specific pools.** Do not let a burst of addition evaluations
   alter Fashion-MNIST wall time, or vice versa, even if both also obey a global
   safety ceiling.
4. **Prefer dedicated remote workers for fixed-time objectives.** If affordable,
   run Fashion-MNIST or other timing-sensitive tasks on isolated, identically
   provisioned workers rather than a thermally variable laptop.
5. **Freeze CPU threads, process priority, power mode, and thermal policy.** Log
   them in the environment receipt.
6. **Record actual contention.** Store queue time, evaluator start/end,
   concurrent jobs, CPU/GPU memory pressure, thermal/power state where
   available, and whether the machine slept during the call.
7. **Separate validity timeout from objective compute.** If a timeout is only a
   safety guard, set it far enough above isolated worst-case runtime that normal
   architecture variation is not censored. If compute is intentionally part of
   the objective, state and measure that explicitly.
8. **Never count queue wait against candidate time.** Preserve the current good
   behavior, and test it end to end under crash/restart.

### 3.2 Freeze one prompt snapshot per campaign, not per trajectory

V1.7/v2.1 currently allow still-unstarted trajectories to pick up later edits.
That is operationally convenient but permits nominally identical C1/C3 runs to
receive different treatment text. A successor should:

1. allow editing until a deliberate `freeze-prompts` launch gate;
2. snapshot one byte-identical ordinary and assumption-changing prompt bundle
   for the entire campaign;
3. record the bundle hash in every run manifest;
4. refuse first start until all scheduled runs point to that same snapshot; and
5. require a new campaign/version for later prompt experiments.

### 3.3 Freeze the actual model service behavior

1. Freeze model identifier, reasoning effort, service tier, sandbox, tool
   policy, Codex CLI version, and provider/API mode for the whole campaign.
2. Do not switch Fast/default tier midway through a confirmatory campaign.
3. Prefer an immutable model snapshot when the provider offers one. Otherwise
   start paired blocks close in wall time and record provider dates so alias
   drift is visible.
4. Add a preflight that renders and dry-checks the exact Codex invocation for
   every condition without consuming an opportunity.
5. Record model/tier/settings both at proposal start and completion, not only in
   the final event.
6. Add a fail-closed check if the provider silently substitutes a model or
   reasoning setting.

### 3.4 Make remote evaluation resumable and idempotent

Transport failure should not automatically discard a completed expensive
training result. For Modal and future remote backends:

1. derive a stable evaluator call ID from campaign/run/opportunity/candidate;
2. atomically record dispatch before the remote call starts;
3. make the remote worker persist a signed/hash-checked result under that ID;
4. let recovery retrieve an already completed result without rerunning it;
5. reject a second execution for the same ID;
6. distinguish “never dispatched,” “running,” “completed but response lost,”
   “worker failed,” and “client interrupted”;
7. charge the evaluator call exactly once; and
8. test laptop sleep, network loss, supervisor death, and Modal client timeout.

This is result retrieval, not a scientifically selective retry.

### 3.5 Add a full campaign health gate

Before launch, verify all of the following in the actual supervisor runtime:

- accelerator allocation and a real tensor operation;
- dataset/cache hashes and read permissions;
- available disk space and inode budget;
- Codex authentication and one non-scientific probe call;
- Modal profile/app/volume/GPU type when relevant;
- prompt, protocol, task, framework, runtime, dependency, and parent hashes;
- per-run locks, thread registry, screen session ownership, and scheduler paths;
- system clock/timezone and sufficient battery/power policy;
- dashboard/health endpoint freshness; and
- a complete crash-recovery smoke test on a disposable campaign.

## 4. Improve the assumption-changing treatment

### 4.1 Preserve the good core of the current prompt

Keep the current open-ended features:

- no fixed mechanism-family menu;
- no declaration that the subject is in an experiment;
- no C0–C3, treatment, benchmark, or checkpoint labels;
- no prohibition on broad categories merely because they are unfamiliar;
- explicit use of prior evidence; and
- a requirement to change representation or computation rather than rename a
  scalar tweak.

### 4.2 Add feasibility without turning the prompt into a recipe (P1)

Candidate direction for the next prompt, to be tested before freezing:

> Identify one load-bearing assumption in the current designs and the strongest
> evidence for and against it. Choose the smallest decisive implementation that
> tests a genuinely different learned mechanism while remaining feasible under
> the fixed training and model constraints. Preserve proven components unless
> changing them is necessary to test the alternative. Explain what result would
> falsify the new approach.

This may reduce Fashion-MNIST timeouts and giant bundled rewrites while leaving
the mechanism space open. It should be piloted blindly against the current
wording before adoption.

### 4.3 Measure realized compliance (P0/P1)

Do not treat an opportunity label as proof of an assumption change. Add a
blinded manipulation-check annotation for every C1/C3 checkpoint and a matched
sample of C0/C2 ordinary proposals:

- old assumption identifiable: yes/no;
- new mechanism actually implemented: yes/no;
- mechanistically distinct from recent lineage: yes/no;
- primarily tuning/pruning/deletion: yes/no;
- cleanly attributable versus bundled: yes/no;
- feasible under the task contract: yes/no; and
- novelty relative to the campaign, scored without condition labels.

Use source deltas, not the model's self-description, as authoritative.

### 4.4 Reconsider the every-tenth-opportunity dose (P1)

1. Use pilot discovery curves to compare every 10, every 20, and a smaller
   number of phase-boundary interventions.
2. Prefer more independent blocks over an automatically longer 200-proposal
   horizon if discovery saturates early.
3. Freeze C1 and C3 to identical checkpoint indices.
4. Consider frozen, block-paired jittered checkpoints so a continuous session
   cannot learn a simple “every ten turns” rhythm. Use the same realized
   schedule for C1/C3 within a block.
5. Do not adapt checkpoint timing to online performance in the main 2×2. An
   adaptive stagnation trigger is a separate research variant.
6. Consider a cooldown rule only if it is fully mechanical and common: for
   example, an assumption-changing checkpoint followed by two ordinary
   exploitation opportunities. This changes the intervention dose and needs a
   new protocol.

### 4.5 Distinguish exploration quality from immediate retention (P1)

An assumption-changing proposal can be scientifically informative even when it
does not immediately beat the parent. Prespecify secondary outcomes:

- Layer-B qualification and novel-cluster yield at the checkpoint;
- valid-but-not-retained mechanism yield;
- probability that the same mechanism produces a later retained descendant;
- immediate and lagged objective change;
- timeout and failure rate;
- source/architecture distance from parent and recent designs; and
- evaluator and token cost per realized mechanism.

Do not change the primary outcome after seeing these analyses.

### 4.6 Candidate intervention variants for separate experiments

- **Reflection-then-edit:** one short non-editing reflection message followed by
  an ordinary edit message. This doubles model calls and is not equivalent to
  the current intervention.
- **Evidence contradiction:** ask the agent to identify a result that the
  current working theory cannot explain before proposing an edit.
- **Counterfactual design:** ask what it would build if the current incumbent
  architecture were unavailable.
- **Constraint inversion:** temporarily prioritize a different bottleneck such
  as inference compute or representation rank while keeping final validity
  unchanged.
- **Stagnation-triggered intervention:** activate after a predeclared number of
  non-improving results rather than by fixed index.
- **Diverse intervention library:** randomly assign one of several frozen
  semantically equivalent prompts within blocks to test wording robustness.
- **Self-critique control:** give C0/C2 an equally long generic reflection prompt
  that does not request assumption change, isolating content from extra
  deliberation/token dose.

Each is a new treatment definition, not a cleanup of v1.7/v2.1.

## 5. Improve the portfolio-memory treatment

### 5.1 Decide which portfolio estimand is intended

Choose and state one of these before implementation:

1. **Whole-system effect:** C2/C3 get more source and therefore more context and
   token cost. This estimates the practical effect of a K=4 portfolio system.
2. **Branch-diversity effect at fixed context:** every condition receives a
   similar evidence/token envelope, but C2/C3 distribute it across live
   branches while C0/C1 receive the same amount from one lineage.

The current protocol mostly estimates the first. Token normalization is useful
but cannot fully remove cognitive effects of longer prompts.

### 5.2 Use a fixed evidence envelope (P1)

For the second estimand:

- cap source-plus-summary evidence to a frozen token budget;
- allocate that budget across K branches in C2/C3;
- use an equal-depth current-lineage history in C0/C1;
- truncate by a deterministic rule, never by model judgment;
- keep metrics and outcome counts parallel across cells; and
- record both original and rendered evidence sizes.

Do not pad with conspicuous fake slots or meaningless prose. Equalize useful
evidence structure, not raw bytes at the cost of a new artifact.

### 5.3 Improve branch quality and diversity (research variants)

- Compare `K=2`, `K=4`, and `K=8` in separate prospective campaigns.
- Compare the current fair-lineage selector with fitness-proportional,
  uncertainty-aware, or novelty-aware selection.
- Compare unconditional fill with a minimum-quality floor before a weak branch
  occupies a slot.
- Preserve one elite slot plus explicit exploratory slots.
- Retain by Pareto front over objective, source novelty, and compute rather than
  objective alone.
- Prevent near-identical candidates from occupying multiple slots using a
  deterministic AST/IR similarity threshold.
- Track lineage age and periodically retire branches with no valid descendants.
- Schedule explicit cross-parent recombination opportunities for portfolio
  cells. Treat this as part of a broader population-system intervention, not
  the same memory-only estimand.
- Test an archive larger than K from which only K evidence items are retrieved.
- Compare full source branches with compact source diffs or architecture IR.

### 5.4 Improve portfolio diagnostics (P1)

Record and analyze:

- branch occupancy over time;
- parent-selection entropy and maximum share;
- lineage depth, breadth, and survival;
- branch source/mechanism distance;
- number of cross-branch ideas reused;
- time a newly filled slot takes to improve;
- fraction of portfolio context actually cited in the proposal;
- prompt tokens per branch and per selected parent; and
- whether intervention checkpoints disproportionately select or destroy weak
  branches.

## 6. Autoresearch-specific improvements

### 6.1 Replace unbounded raw transcript memory (P0)

The strongest candidate for Autoresearch vNext is a bounded-session design:

1. run a fresh Codex conversation at a fixed cadence, preferably every
   opportunity or every small frozen phase;
2. provide the current source, current public metrics, and a deterministic
   state capsule;
3. include exactly the controller memory permitted by C0–C3;
4. exclude raw tool logs, old filesystem dumps, routine status messages, and
   superseded source; and
5. preserve the original transcript privately for audit but do not replay it.

This would greatly reduce token growth and make the portfolio factor cleaner.
It also moves farther from the ecological form of a long-lived Autoresearch
session, so the existing continuous mode should remain a separate framework
variant rather than being retroactively reinterpreted.

### 6.2 Candidate state-capsule contents

Use a deterministic controller renderer, not an LLM-generated summary, for:

- current source and public verified metrics;
- last outcome on the selected lineage;
- compact hypotheses/results for a frozen number of earlier outcomes;
- retained alternative source or diffs only in portfolio cells;
- known failure categories in subject-level language;
- agent-authored short research notes if they fit a frozen size limit; and
- the current task and output contract once per new conversation.

Hash the capsule and save it with the prompt. Never include internal fitness,
condition, opportunity number, remaining horizon, tokens, selection counts, or
private evaluator fields.

### 6.3 If continuous sessions are retained, bound them explicitly (P1)

- Start a new session at frozen phase boundaries and carry forward only the
  state capsule.
- Set and log a maximum context size before launch.
- Detect compaction or provider-side history truncation and record its boundary.
- Remove routine tool output from resumed context when the CLI supports it.
- Prevent agents from polling training; keep evaluation wholly outside the
  subject turn as it is now.
- Cap final summaries and routine command output without capping useful
  reasoning.
- Report cached input separately from newly ingested input and from output.
- Add a context-growth smoke test covering all 200 opportunities.

### 6.4 Make the memory contrast honest (P0/P1)

In a continuous transcript, C0/C1 can remember rejected source, earlier
mechanisms, and old results even when the controller shows one incumbent. For a
clean memory experiment either:

- use fresh/bounded sessions in all four conditions; or
- add conversation memory as a third randomized factor in a separate 2×2×2
  experiment.

Do not describe the current continuous C0/C2 contrast as memory versus no
memory.

### 6.5 Improve direct-edit reliability without hidden search

1. Provide a protected, non-scoring static checker for syntax, imports,
   editable-path compliance, model-interface shape, and parameter-ceiling
   feasibility.
2. Make clear which libraries are unavailable in the editing sandbox so the
   agent does not repeatedly invoke an interpreter lacking PyTorch.
3. Allow only checks of the one submitted implementation; do not expose the
   validation set or permit batches of alternatives.
4. Run the same deterministic preflight controller-side before expensive
   training and return a sanitized reason on the next opportunity.
5. Treat preflight failure as a consumed proposal, not a retry.
6. Preserve an ordinary local Git diff so the agent can inspect its one change.

### 6.6 Improve Autoresearch provenance

- Parse or post-process the final summary into old assumption, new mechanism,
  hypothesis, intended edit, expected effect, risk, and cited evidence.
- Do not invalidate an otherwise valid source submission merely because prose
  labels are absent.
- Record source diff statistics and architecture IR independently of the
  agent's claims.
- Detect whether a resumed session ID, cwd, prompt snapshot, or editable source
  unexpectedly changes.
- Store explicit context/compaction boundaries for every call.
- Add a deterministic transcript audit that checks for forbidden path,
  benchmark, condition, and checkpoint leakage.

### 6.7 Add an ecological-validity companion (research variant)

The controlled one-edit/one-evaluation adapter is not identical to the original
Karpathy workflow in which a long-lived agent can inspect results and keep
working. A separate companion stratum could run a more authentic Autoresearch
loop with a fixed wall-clock or evaluator budget. It should not be pooled with
the controlled factorial stratum, but agreement between them would strengthen
external validity.

## 7. OpenEvolve-specific improvements

### 7.1 Preserve bounded ephemeral proposals (P0/P1)

This is one of v2.1's strongest design choices. It keeps per-proposal token cost
roughly stable, prevents accidental transcript memory, and makes C0–C3 evidence
controller-auditable. Retain it unless conversation mode itself becomes an
explicit research factor.

### 7.2 Replace “last 12” with a fixed informative-memory policy (P1)

Recency alone forgets old decisive failures and encourages repetition. A
deterministic, condition-common evidence selector could include:

- the most recent result;
- the current incumbent's provenance;
- the best prior success;
- the most informative failure for each recent self-described mechanism;
- a frozen number of diverse older results selected by source/IR distance; and
- recent infrastructure failures only as a generic “no result” message.

Keep the rendered evidence under a fixed token cap. Freeze the selector before
launch and never use Layer B labels online.

### 7.3 Normalize mechanism memory without a restrictive menu (P1/P2)

- Keep free-form subject labels.
- Create controller-side semantic fingerprints from source deltas and
  architecture IR.
- Use those fingerprints to flag likely repetitions in process logs.
- If repetition feedback is shown online, describe the earlier source/result
  rather than assigning a fixed family label.
- Keep human-blinded Layer B authoritative.
- Evaluate false merges/splits on a held-out manually annotated sample before
  using semantic fingerprints for retention or retrieval.

### 7.4 Do not gate candidate validity on prose metadata (P1)

If a patch applies, passes source checks, trains, and qualifies, missing
`MECHANISM` or `EVIDENCE` prose should not necessarily erase the scientific
candidate. Better options are:

- evaluate the patch and mark metadata incomplete;
- recover hypothesis/mechanism post hoc from the response and source delta; or

- require structured metadata only for a separate provenance-compliance
  outcome.

The choice must be frozen prospectively. Source validity and response-format
compliance should remain distinct failure categories.

### 7.5 Improve source presentation only if it saves real tokens

Patch-format failure is already rare. Potential efficiency variants include:

- materialize selected and reference source in an opaque read-only cwd so Codex
  can use normal `rg`/`sed` tools instead of embedding every file in the prompt;
- provide deterministic compact diffs for references while keeping the full
  selected parent;
- include line anchors or file hashes for exact patch application;
- keep atomic multi-file application and exact-once matching; and
- test token use and patch success before adopting a new representation.

Do not add another response schema merely for aesthetic cleanliness.

### 7.6 Add a native OpenEvolve companion (research variant)

The controlled adapter deliberately replaces native database sampling,
islands, population updates, and retention because those overlap C0–C3. A
separate ecological-validity campaign should run pinned native OpenEvolve end
to end under the same task/evaluator budget. Compare it descriptively with the
controlled adapter; never call the current v2.1 system full native OpenEvolve.

### 7.7 OpenEvolve population variants

- Compare mutation-only with frozen crossover opportunities.
- Compare one population with islands and migration.
- Compare current lineage-local strict improvement with native quality-diversity
  retention.
- Test novelty-aware prompt retrieval independently of novelty-aware retention.
- Test whether reference source, reference summaries, or both produce the
  portfolio effect.
- Hold total prompt tokens fixed when comparing population sizes.

## 8. Task and evaluator improvements

### 8.1 Use paired repeated training seeds (P1)

One stochastic training run can make strict retention noisy and create a
winner's curse. Options, in increasing cost:

1. keep one common run seed online but re-evaluate retained incumbents on a
   second sealed seed;
2. evaluate parent and child with paired common random numbers in one evaluator
   call;
3. rank on the mean/median of a small frozen seed set; or
4. use sequential confirmation only when a child is within a predeclared margin
   of its parent.

The same rule and seed set must apply to all conditions. If seed averaging is
too expensive, at least estimate repeatability on calibration and final
incumbents.

### 8.2 Separate qualification, ranking, and final generalization

- Use a public development metric for online feedback.
- Use an evaluator-private confirmation split/seed for retention when feasible.
- Keep Layer C sealed and unavailable until all trajectories finish.
- Never use Layer C to select or resume candidates.
- Report adaptive public-validation overfitting by plotting Layer A versus
  Layer C as a function of opportunity.

### 8.3 Addition-specific improvements

- Verify across several disjoint operand seeds and aggregate exact accuracy.
- Add length generalization or boundary/carry-heavy sealed audits if the task
  contract permits them prospectively.
- Expand anti-solver audits with behavioral counterfactuals, weight
  perturbation, attention intervention, decoder integrity, and source/bytecode
  checks.
- Test false-positive and false-negative rates of the learned-attention
  requirement on known legitimate and cheating fixtures.
- Decide whether “transformer with causally necessary attention” or “learned
  sequence model” is the intended class. A broader class should be a separate
  task version.
- Report unique trainable scalars, serialized model bytes, inference FLOPs,
  latency, activation memory, training steps, and energy alongside the primary
  deduplicated-parameter objective.
- Define weight tying and procedural parameter generation explicitly so the
  size metric cannot be gamed by representation tricks that do not reduce real
  model description length.
- Consider a Pareto analysis of parameters versus accuracy margin above 99%,
  while keeping the frozen threshold objective primary.
- Calibrate whether 5,000 steps gives every admissible architecture a fair
  chance; if not, define a compute budget rather than silently increasing steps
  for selected candidates.

### 8.4 nanoGPT-specific improvements

- Create a prospective genuinely sealed validation shard or corpus segment for
  Layer C; the current repeat is only replication.
- Record actual H100 subtype, driver, CUDA, PyTorch, clocks if available,
  worker ID, compilation time, measured training time, steps, tokens, MFU, and
  peak memory.
- Verify that startup/compile exclusions cannot be exploited by moving useful
  training outside the measured window.
- Freeze validation frequency and accounting in protected code.
- Add a protected check for data leakage, cached learned state, pretrained
  weights, and validation-set training.
- Use identical warm/cold container policy across conditions.
- Consider a cheaper proxy phase only as a multi-fidelity research variant;
  confirm every retained/final design on the full five-minute objective.
- Report quality per GPU-second and estimated dollar cost in addition to
  `val_bpb`.
- Audit whether architecture-dependent compile failures or unsupported kernels
  disproportionately affect assumption-changing proposals.

### 8.5 Fashion-MNIST-specific improvements

- Reduce MPS concurrency to the isolated calibrated limit.
- Replace the 90-second wall guard with a limit justified by isolated runtime,
  or make inference/training compute an explicit objective.
- Log timeout candidates' partial metrics separately from valid results without
  retaining them.
- Prevent repeated adaptive overfitting to the same 10,000-image validation set
  with a prospective private confirmation split or seed-averaged training.
- Keep the official test split sealed for Layer C.
- Decide whether test-time augmentation, ensembling, EMA, logit scaling, and
  calibration are intended mechanisms or unrestricted compute loopholes.
- Freeze or price inference views if the research object is model architecture
  rather than unbounded inference compute.
- Report parameter count, MACs/FLOPs, inference views, latency, peak memory, and
  training time with accuracy.
- Consider a bi-objective or constrained-compute successor task; do not change
  the current accuracy-first objective in place.
- Calibrate the 250k ceiling against the parent and mechanism space. A hard
  ceiling causes many proposals to cluster just below it and may reward
  parameter bookkeeping rather than discovery.
- Add corruption/shifted-image Layer C sets only if frozen before collection.

### 8.6 Multi-fidelity evaluation (research variant)

A deterministic cheap screen can save compute, but it can also bias against
slow-learning mechanisms. If tested:

- use the same successive-halving schedule for all cells;
- predeclare promotion thresholds independent of condition;
- count every started candidate as a proposal;
- distinguish screen calls from full evaluator calls;
- require full evaluation before online retention; and
- audit which mechanism families the screen falsely rejects.

## 9. Budgets, replication, and scheduling

### 9.1 Buy independent runs before extremely long trajectories (P1)

The experimental unit is a trajectory. Once discovery curves flatten, another
100 proposals in the same transcript add less inferential value than another
paired block. Use current curves to simulate designs such as:

- 8–12 blocks × 100 proposals;
- 6–8 blocks × 150 proposals; or
- a short pilot horizon followed by a separately frozen confirmatory horizon.

Choose based on expected qualified-cluster yield, within-block variance,
reviewer workload, tokens, and evaluator cost. Do not treat thousands of
proposals as thousands of independent samples.

### 9.2 Predeclare stopping and incompleteness

- No performance-based early stop in the confirmatory design.
- Allow only hard proposal/evaluator/compute safety limits that are common by
  condition.
- Predeclare how partial trajectories appear in primary and sensitivity
  analyses.
- Report every provider, timeout, infrastructure, preflight, and
  nonqualification failure.
- Never delete/retry scientifically inconvenient attempts.
- Use idempotent retrieval rather than rerunning a completed remote evaluation.

### 9.3 Balance timing and provider drift

- Launch C0–C3 of each block in the same narrow time window.
- Interleave blocks across tasks/frameworks rather than completing one
  condition or framework months earlier.
- Use a condition-blind evaluator queue with deterministic or randomized
  within-block ordering.
- Record queue order and actual overlap.
- Prevent a single run from monopolizing provider or evaluator capacity.
- Include launch date/time and service incidents in analysis receipts.

### 9.4 Model and task replication

- Replicate the 2×2 on at least one additional model family or model snapshot.
- Treat model as a separate stratum or hierarchical level.
- Prefer a few complete high-quality task strata over many incomplete ones.
- Add tasks that differ in mechanism space and evaluator regime, not just
  another image dataset with the same loopholes.

## 10. Outcomes and statistical analysis

### 10.1 Strengthen Layer B review (P0/P1)

1. Freeze a task-specific rubric before packet export.
2. Use at least two genuinely independent blinded reviewers.
3. Build an annotation tool that shows parent/candidate diff, source, and claim
   without revealing task score, run, condition, or opportunity.
4. Randomize packet order and duplicate a hidden subset to measure reviewer
   consistency.
5. Measure qualification agreement and pairwise co-clustering agreement.
6. Adjudicate while still blinded and hash the adjudicated file before opening
   the condition mapping.
7. Ask reviewers to guess condition after adjudication as a blinding check.
8. Preserve notes explaining each merge/split.
9. Predeclare how multi-mechanism bundles are handled.
10. Keep exact/source-equivalent dedup deterministic before human review, but
    do not let an automatic semantic cluster replace human adjudication.

### 10.2 Add mechanism-quality dimensions

Distinct cluster count treats a weak one-off and a strong repeatable mechanism
equally. Keep that primary if desired, but add blinded secondary ratings:

- novelty within the campaign;
- causal attribution/cleanliness;
- falsifiability;
- immediate task value;
- repeatability in descendants or seeds;
- computational feasibility;
- generalization to Layer C; and
- whether it combines known components versus changes the learned computation.

### 10.3 Add trajectory-level secondary outcomes

- cumulative unique-cluster discovery curves;
- time/proposal/token/evaluator-hour to first and last novel cluster;
- clusters per million priced-token units, evaluator-hour, and dollar estimate;
- final Layer A and Layer C objective;
- valid, qualified, retained, timeout, and infrastructure rates;
- immediate and lagged intervention yield;
- lineage depth/breadth/survival and parent-selection entropy;
- portfolio source citation/reuse;
- source/IR diversity and mechanism entropy;
- tuning versus mechanism-change share;
- repeated mechanism rate;
- hypothesis calibration: predicted direction versus actual result; and
- performance–diversity Pareto fronts.

### 10.4 Improve inferential planning

- Simulate power/interval width from pilot run-level cluster counts before
  choosing block count.
- Use block-paired contrasts and show every raw run.
- For counts, predeclare Poisson/negative-binomial or a robust alternative and
  check overdispersion/zero inflation.
- Do not bootstrap proposals as if independent.
- Keep each task × framework × protocol result primary at its own level.
- Use hierarchical synthesis only after publishing raw strata.
- Predeclare missing-run handling and sensitivity analyses.
- Correctly label transition timing, mechanism families, and mediation as
  secondary/exploratory unless separately powered.
- Report effect sizes and compatibility intervals, not only significance.

## 11. Instrumentation and reproducibility

### 11.1 Token and monetary accounting

- Record input, cached input, output, and reasoning output per Codex item and
  per proposal.
- Store the exact model/tier and a versioned price table separately from raw
  token counts.
- Report both list-price-equivalent cost and actual quota/credit usage when
  available.
- Attribute setup, proposal generation, tool calls, evaluator waiting, and
  routine polling separately.
- Detect unpriced or missing-token events.
- Keep cached tokens distinct; never double-count them as additional input.

### 11.2 Time accounting

- Store active agent time, evaluator queue time, evaluator runtime, remote
  worker time, pause time, sleep time, and wall time separately.
- Derive pause intervals from lifecycle events, not gaps alone.
- Record laptop sleep/wake and supervisor downtime as infrastructure intervals.
- Use monotonic clocks for durations and UTC timestamps for provenance.
- Mark work that overlaps across runs so campaign “total time” is not obtained
  by summing concurrent intervals.

### 11.3 Candidate and environment provenance

- Content-hash every prompt, response, editable file, protected task file,
  dependency lock, evaluator image, and calibration artifact.
- Record parent/child source diffs and a normalized architecture IR.
- Store GPU/CPU/OS/driver/Python/PyTorch/Codex versions with every campaign.
- Verify source immutability during training and final verification.
- Make run state and lifecycle schemas versioned and migration-tested.
- Add a read-only `audit-campaign` command that checks event/state/candidate
  consistency after crashes and before Layer B.
- Verify there is exactly one completed record or one explicitly active record
  per opportunity.
- Detect duplicate writer PIDs, stale locks, reused conversation IDs, and
  result files without matching dispatch records.

### 11.4 Health monitoring

- One command should summarize supervisors, desired/actual state, heartbeat,
  active opportunity, last event age, queue ownership, evaluator progress,
  token usage, best objective, failure streak, and disk usage.
- Classify “queued,” “agent thinking,” “evaluating,” “cooperatively pausing,”
  “paused,” “recoverable interruption,” and “stuck” separately.
- Alert only on actionable conditions; routine healthy polling should not
  consume subject-model tokens.
- Run the dashboard as a supervised local service or document one reliable
  restart command.
- Keep the dashboard strictly read-only and outside scientific state.

### 11.5 Archive and release

- Freeze a manifest and hashes before moving data.
- Separate public artifacts from secrets, Codex auth, private Layer-B mapping,
  and sealed test data.
- Archive complete event logs, Codex item logs, evaluator outputs, candidate
  snapshots, prompt snapshots, amendments, environment receipts, and review
  files.
- Provide a small replay tool that reconstructs every online decision from
  immutable events.
- Generate a machine-readable data dictionary and schema version.
- Test the release from a clean machine before paper submission.

## 12. Prompt and subject-boundary improvements

### 12.1 Keep subjects unaware of the experiment

Retain the v1.7/v2.1 artifact-clean boundary. Subjects should see an ordinary
optimization job, not protocol names, C0–C3 labels, “assumption checkpoint,”
benchmark names where they create demand effects, hidden evaluator internals,
budgets, horizons, seeds, controller paths, or prior campaign reports.

### 12.2 Reduce unnecessary instruction load

- State the task contract once per fresh conversation.
- In later messages send only new evidence, current source/state, and the active
  direction.
- Remove duplicated response instructions and source copies.
- Keep one authoritative work-boundary section.
- Use short sanitized evaluator reasons rather than raw stack traces.
- Do not include fake paths, empty slots, internal IDs, selection counts, or
  remaining resources.
- Audit rendered prompts, not only templates.

### 12.3 Avoid new restrictions that narrow discovery

Do not add:

- a fixed mechanism-family list;
- a rule allowing only one tiny file edit;
- a blanket prohibition on revisiting old ideas when new evidence justifies it;
- condition-specific tools or compute;
- hints about expected parameter targets or known successful designs;
- examples copied from ongoing trajectories; or
- language asking the subject to behave differently because it is being
  studied.

Feasibility and attribution guidance should shape experimental discipline, not
predetermine the mechanism.

## 13. Suggested coherent successor bundles

These are recommendations, not implementations.

### 13.1 Minimal Autoresearch successor

The highest-value coherent Autoresearch vNext bundle would be:

1. bounded/fresh conversations with a deterministic state capsule;
2. one campaign-wide frozen prompt snapshot;
3. a fixed model snapshot/tier and no live tier changes;
4. task-isolated evaluator concurrency calibrated for result equivalence;
5. idempotent remote result retrieval;
6. the current open-ended intervention plus a concise feasibility/falsification
   clause, validated in a non-paper pilot;
7. blinded realized-intervention compliance annotations;
8. more paired blocks and a shorter horizon selected from discovery curves;
9. improved Layer B tooling and preregistered secondary outcomes; and
10. full context/token/queue/sleep provenance.

The bounded-memory change is large enough that this successor must be analyzed
separately from continuous v1.7. If ecological Autoresearch continuity is a key
question, run continuous v1.7-like behavior as a companion stratum.

### 13.2 Minimal OpenEvolve successor

The highest-value coherent OpenEvolve vNext bundle would be:

1. preserve fresh ephemeral proposal calls and strict atomic patches;
2. one campaign-wide frozen prompt snapshot;
3. fixed model/tier and balanced launch timing;
4. isolated/calibrated evaluator concurrency and idempotent remote retrieval;
5. replace pure last-12 recency with a fixed informative evidence selector
   under a token cap;
6. treat missing prose metadata separately from source/model validity;
7. retain free-form mechanisms but add controller-side semantic repeat
   diagnostics;
8. add the same intervention feasibility/manipulation checks as Autoresearch;
9. increase paired blocks before increasing trajectory length; and
10. run pinned native OpenEvolve only as a separate ecological-validity
    companion.

### 13.3 Items to keep out of the first successor

Do not initially bundle K sweeps, adaptive interventions, novelty-aware
retention, crossovers, multi-fidelity training, multiple model families, a new
task objective, and native-framework execution into the same version. Each can
be valuable, but together they would make it impossible to identify which
change fixed or caused an effect.

## 14. Decisions required before implementation

1. Is the memory estimand the whole practical portfolio system or branch
   diversity under matched context?
2. Should Autoresearch prioritize ecological continuous-session fidelity or a
   clean bounded-memory factorial contrast?
3. Is Fashion-MNIST runtime merely a safety guard or part of the objective?
4. How many isolated MPS evaluations are empirically equivalent to one?
5. Should the intervention remain every ten opportunities, become less
   frequent, or be tested as a separate dose experiment?
6. Is 200 proposals still worth more than additional paired blocks after
   accounting for discovery saturation and annotation cost?
7. Should retained candidates require one seed, paired confirmation, or a
   frozen seed set?
8. Which task gets a genuinely sealed Layer C in the next version?
9. Is missing subject prose a validity failure or only a provenance outcome?
10. Which exact model snapshot/tier and hardware backend can remain stable for
    the full campaign?

Until those choices are made, this document should remain a backlog and no
successor protocol should be labeled frozen.
