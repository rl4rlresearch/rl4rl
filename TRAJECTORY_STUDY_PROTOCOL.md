# Protocol: How Autonomous Research Systems Search

**Version:** 1.0

**Status:** executable protocol; real-data analysis not yet run

**Benchmark:** AdderBoard

**Systems:** OpenEvolve, Autoresearch, TTT-Discover

**Independent unit:** one complete, independently launched search run

## 1. Research objective

The operational task is:

\[
\min \operatorname{parameters}(m)
\quad\text{subject to}\quad
\operatorname{exact\_match}(m) \ge 0.99
\]

The scientific objective is to characterize how autonomous research systems
search that space. Endpoint parameter count is necessary but insufficient: two
systems can reach similar endpoints through very different proposal,
evaluation, acceptance, rollback, diversification, and stopping behavior.

The central hypothesis is that current systems disproportionately make
**ontology-preserving** local compression moves—changing width, depth, head
count, normalization, or parameter sharing within a familiar architecture—and
rarely make **ontology-changing** moves that alter how the task is represented
or computed, such as parametric token representations, different positional
algebra, or task-structured projections.

The benchmark is a controlled apparatus, not a proxy for science as a whole.
The permitted conclusion is about these systems under the frozen AdderBoard
conditions. The study must not claim that all autonomous research agents, all
models, or scientific discovery generally share an observed behavior.

## 2. Questions and hypotheses

### Primary questions

1. What fraction of attempted edits is ontology-preserving,
   ontology-changing, mixed, or unclassified?
2. How often are ontology-changing attempts accepted and how often do they
   produce a qualifying ≥99% model?
3. How quickly does each run improve its qualifying parameter frontier?
4. Does edit-family diversity collapse over the run, and how often does the
   search revisit an already explored architecture fingerprint?
5. What evidence accompanies stopping, and does a run claim a floor while a
   frozen external frontier remains below its best candidate?

### Secondary questions

- Which edit families dominate each run?
- How long are uninterrupted ontology-preserving streaks?
- How often do rollbacks and invalid candidates occur?
- Do invalid candidates become parents of later valid candidates (a possible
  “invalid/cheating anchor” pathway that requires qualitative inspection)?
- How does lineage breadth/depth differ across paradigms?

### Directional hypotheses

- **H1:** ontology-preserving edits outnumber ontology-changing and mixed edits.
- **H2:** edit-family diversity declines in later trajectory windows.
- **H3:** ontology-changing edits have a lower acceptance rate but may account
  for a disproportionate share of large frontier improvements.
- **H4:** runs stop above the known external frontier, sometimes while using
  language consistent with local optimality.

These hypotheses are mechanistic descriptions, not licensed causal comparisons
unless system assignment, budget, prompts, tools, hardware, and starting state
were prospectively controlled.

## 3. Study design

### Observational core

The initial paper is a comparative trajectory case series. It analyzes complete
native histories from OpenEvolve, Autoresearch, and TTT-Discover. The study does
not retrofit a common controller around them and then claim to have reproduced
their native paradigms.

For each run, retain:

- the initial program/model and its hash;
- every proposal, commit, mutation, rollout, evaluation, decision, rollback,
  and stop record available;
- prompts, feedback, model/API configuration, tool access, compute budget,
  evaluator version, seeds, and wall-clock limits;
- candidate source or patch hashes and lineage identifiers;
- accuracy, parameter count, validity, failure, and selection metadata;
- the reason for missing or excluded events.

Curated leaderboards or winner-only histories are not complete trajectories and
cannot support the primary analysis.

### Optional controlled extension

If the systems can later be run independently under matched conditions, freeze
the starting candidate, evaluator, task split, accuracy threshold, generator
model, provider effort, token ceiling, wall-clock/accelerator budget, number of
proposal opportunities, and stopping policy. Repeat launches with predeclared
independent seeds. Do not treat multiple candidates within one launch as
replicates.

### Independence

An independent run requires a fresh launch whose stochastic state is not
derived from another analyzed run and whose agent memory/archive is not reused.
Forks, resumed runs, or branches sharing an archive must be grouped under one
run family and cannot be counted as independent without a predeclared model of
that dependence.

## 4. Qualification and frontier

A candidate qualifies when all of the following are recorded by the frozen
evaluator:

1. it is executable and valid under the task rules;
2. exact-match accuracy is at least the manifest threshold (default `0.99`);
3. parameter count is a positive integer from the agreed counting procedure.

The within-run frontier improves whenever a newly evaluated qualifying
candidate has fewer parameters than every earlier qualifying candidate in that
run. The manifest may freeze an external published/official frontier for
distance calculations. The default template uses 36 parameters because that is
the frontier stated in the supplied project materials; it must be verified and
cited before publication rather than silently assumed from this repository.

No score normalization, percent/fraction guessing, or missing-validity
imputation is allowed.

## 5. Normalized event and lineage contract

Every native record maps to a typed event containing:

- study run, source, event, candidate, and parent identities;
- paradigm and strictly ordered sequence index;
- event kind and decision;
- optional timestamp, accuracy, parameter count, validity, description,
  fingerprint, and stop claim;
- untouched adapter-specific metadata;
- raw source identifier, zero-based record index, and exact source SHA-256.

The validator rejects duplicate event/sequence identities, cross-paradigm runs,
parents that appear after children, non-terminal stop records, non-finite or
out-of-range accuracy, invalid parameter counts, metrics without a validity
flag, unknown schema fields, missing files, path traversal, and hash mismatch.

The normalization layer should lose as little information as possible. New
native fields belong in `metadata` until a protocol amendment promotes them to
the shared schema.

## 6. Architecture edit codebook

### Boundary classes

- **ontology_preserving:** changes capacity, arrangement, or regularization
  within the same computational account of the task.
- **ontology_changing:** changes the primitive representation, information
  path, positional algebra, task decomposition, or computational account.
- **mixed:** contains separable preserving and changing components whose effects
  cannot be assigned from the recorded evaluation.
- **unclassified:** evidence is insufficient. This is missingness, not a local
  edit by default.

The classification concerns the parent-to-child delta, not whether the child
looks unusual in isolation. Parameter magnitude is not evidence of a boundary
change.

### Edit families

`scale`, `depth`, `width`, `attention`, `embeddings`, `positional`,
`normalization`, `feedforward`, `parameter_tying`, `tokenization`, `curriculum`,
`optimizer`, `verifier`, and `other`.

Each edit receives one primary family representing its most structurally
consequential component. Rationale text must name the concrete delta. The full
machine-readable rules are frozen in
`architecture_discovery/trajectory_annotation_codebook.yaml`.

## 7. Annotation procedure

1. Prepare parent/child diffs and descriptions with system identity and outcome
   hidden when feasible.
2. Two coders label every parented event independently.
3. Reveal coder labels only after both are frozen.
4. If family or boundary differs, a third adjudicator chooses the final label
   from the evidence; adjudication does not overwrite raw coder records.
5. Report joint exact agreement and Cohen’s κ separately for family and
   boundary class.
6. Inspect disagreements qualitatively and report unclassified cases.

Keyword hints may help route examples but cannot become final labels. This
prevents the code from “discovering” ontology changes through vocabulary that
was built into the analysis.

## 8. Predeclared run-level metrics

The executable pipeline calculates:

- evaluated, valid, accepted, and qualifying candidate counts;
- initial/best qualifying parameter counts and external-frontier gap;
- qualifying frontier progression and improvement count;
- edit-family and boundary-class counts;
- Shannon edit-family entropy and normalized entropy;
- ontology-changing/mixed attempt, acceptance, and qualification rates;
- overall acceptance, invalid, rollback, and revisit rates;
- architecture-fingerprint coverage and unique fingerprints;
- longest ontology-preserving streak;
- sequence of first ontology-changing edit, first qualifying candidate, and
  first frontier improvement;
- immediate invalid-parent to valid-child count;
- recorded stopping claim and a rule-based `premature_frontier_claim` flag.

The stopping flag is descriptive. It activates only when stop text contains a
predeclared floor/optimality phrase and the run’s best qualifying parameter
count exceeds the frozen external frontier. It is not a mind-reading measure.

Rolling diversity is the number of distinct edit families in a five-edit window
divided by the available window length. Sensitivity analyses should use windows
of 3 and 10 without replacing the primary five-edit view.

## 9. Statistical analysis

Primary summaries are per run and per paradigm. Report the distribution of
run-level values and show individual runs. Candidate-level denominators may
describe a run but may not inflate the sample size for system comparisons.

With fewer than five independent runs per paradigm, report no confidence
intervals or hypothesis tests. With five or more, this implementation still
emits descriptive results only; any hierarchical, permutation, or bootstrap
model must be preregistered in an amendment specifying its estimand, dependence
structure, missing-data policy, multiplicity policy, and minimum effect of
interest.

Do not rank systems as intrinsically superior if budgets or observation quality
differ. Stratify or label unmatched conditions. Report both macro averages
(equal run weight) and, only when scientifically useful, within-run rates.

## 10. Exclusions and missing data

Exclude a run from a stated metric only for a predeclared reason:

- corrupt or incomplete native history;
- unverifiable source hash;
- unknown candidate ordering or lineage needed by that metric;
- evaluator incompatibility that makes qualification incomparable;
- annotation coverage failure.

Keep the run visible in the data-quality table. Do not replace missing
fingerprints, accuracy, parameters, validity, or labels with favorable values.
The pipeline reports null denominators explicitly.

Infrastructure failures belong in the trajectory when they affected the
agent’s observed search. Clearly external outages may be described separately,
but removing them requires a frozen rule applied blind to system outcome.

## 11. Sensitivity and qualitative analyses

Predeclared sensitivity views:

- accepted edits only versus all evaluated edits;
- exclude `mixed` versus combine it with ontology-changing;
- earliest evaluation per fingerprint versus all evaluations;
- frontier at equal proposal, token, wall-clock, and accelerator budgets when
  all systems expose those measures;
- runs with complete fingerprints versus all runs;
- external-frontier value omitted versus included.

Qualitative case studies may examine major frontier jumps, failed
ontology-changing moves, rollbacks, invalid-parent chains, and stopping
rationales. Select cases by frozen rules (for example, top two absolute frontier
jumps per run), not narrative convenience.

## 12. Reproducibility and output contract

The manifest freezes every native trajectory and annotation file by SHA-256.
Analysis refuses to overwrite an existing output directory. The result bundle
contains normalized JSONL, run/paradigm JSON and CSV, annotation agreement,
lineage CSV/DOT, three SVG figures, a Markdown report, and provenance with every
input and output digest.

The synthetic smoke run tests software behavior only. Its numbers must never be
copied into a paper results section.

## 13. Launch checklist

Before real analysis:

- [ ] Complete native trajectories obtained for all declared runs.
- [ ] Prompts, tools, budgets, seeds, evaluator, starting point, and stop rules
  documented per run.
- [ ] Independence/fork relationships documented.
- [ ] Accuracy and parameter-count procedures made comparable or differences
  explicitly stratified.
- [ ] External frontier verified and cited, or set to `null`.
- [ ] Source files frozen and manifest hashes verified.
- [ ] Annotation examples calibrated without using study outcomes.
- [ ] Two independent coders complete every edit.
- [ ] Disagreements adjudicated; reliability reviewed before interpreting
  between-system patterns.
- [ ] Validation script passes with no `--allow-unannotated` flag.
- [ ] Output directory is new and analysis command/report are archived.

At repository creation time, the original real trajectories are absent. The
scientific study is therefore **data-blocked**, not complete and not negative.
