# Paper notes, literature synthesis, and reporting checklist

Last literature check: 2026-08-20. Verify titles, versions, author lists, and
workshop formatting again at submission time.

## 1. One-sentence contribution

We run a controlled 2×2 factorial experiment that separates portfolio memory
from scheduled assumption-changing interventions in autonomous ML research,
using identical task/evaluator/budget infrastructure and blinded
mechanism-cluster review across two proposal interfaces and two qualitatively
different ML tasks.

The strongest claim is causal and narrow:

> Within a fixed task × framework × model × budget stratum, changing the
> controller’s memory or scheduled proposal policy changes the number and type
> of valid mechanism families produced.

Do not claim that one framework is universally better, that clusters equal
scientific discoveries, or that the study proves general autonomous-science
capability.

## 2. Why this study is useful relative to existing work

### FML-bench

[FML-bench](https://arxiv.org/abs/2605.17373) evaluates autonomous ML engineering
strategies across 18 tasks and 10 domains and emphasizes separating search
strategy from infrastructure. Its process analysis reports that greedy search
is often surprisingly competitive, while tree/evolutionary methods can help
when improvements are sparse; early convergence and focused effort correlate
with performance. Its public implementation is
[qrzou/FML-bench](https://github.com/qrzou/FML-bench).

Our study uses that separation as an engineering principle but asks a different
causal question: hold the proposal interface and infrastructure fixed while
intervening on memory and scheduled assumption changes. Mechanism diversity,
not final score alone, is primary.

### Heuresis

[Heuresis](https://arxiv.org/abs/2606.25198) decomposes autonomous research into
ideator, executor, grader, auditor, memory, and strategy modules and compares
Greedy, MAP-Elites, Go-Explore, Islands, Curiosity, and Omni across thousands of
scored runs. It highlights that genuinely novel ideas are rare and reward
hacking is common. Code is at
[a-antoniades/Heuresis](https://github.com/a-antoniades/Heuresis).

Our controller is compatible with that modular view. C0/C1 resemble one active
lineage; C2/C3 expose a small explicitly controlled portfolio. The sealed
mechanism review and protected evaluator respond directly to novelty scarcity
and reward-hacking concerns.

### Karpathy Autoresearch

[Karpathy Autoresearch](https://github.com/karpathy/autoresearch) provides a
minimal single-GPU research loop: edit `train.py`, train for a fixed five-minute
budget, and minimize validation bits per byte. It demonstrates the practical
power of a simple greedy direct-edit loop and provides the second task used
here. We use its task and direct-edit proposal interface, not its unmodified
long-lived program prompt or controller.

### OpenEvolve

[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) provides
an AlphaEvolve-inspired evolutionary prompt/diff/database implementation. We
reuse its prompt sampler and SEARCH/REPLACE mutation representation. Native
population selection and retention are replaced by the shared factorial
controller so they cannot confound the memory treatment.

### Trace-based evaluation and long horizons

[EvoTrace](https://arxiv.org/abs/2605.20086) argues that a final score can hide
known-solution recombination, tuning, overfitting, and weak causal attribution;
the trajectory itself must be replayed and analyzed. Our content-addressed
candidates, parent links, prompt hashes, per-opportunity usage, and sealed
parent/candidate packets support that process view.

[Long-Horizon Autonomous Architecture Research](https://arxiv.org/abs/2608.01995)
reports that greedy controllers induce hill climbing and motivates diversified
search, forks, and budgeted high-risk proposals. C2/C3 and the scheduled
transition intervention are controlled tests of two such remedies.

## 3. Research questions and hypotheses

### RQ1 — portfolio memory

Does exposing and maintaining `K=4` live lineages increase distinct qualified
mechanism clusters per run?

- **H1:** `(C2 + C3)/2 > (C0 + C1)/2`.
- Mechanism: preserved alternatives reduce premature convergence and let the
  agent reuse evidence from approaches that are not the global incumbent.
- Countermechanism: extra context dilutes attention, consumes input tokens, and
  encourages shallow recombination.

### RQ2 — assumption-changing checkpoints

Do frozen prompts at opportunities 20/40/60/80 increase distinct clusters?

- **H2:** `(C1 + C3)/2 > (C0 + C2)/2`.
- Mechanism: explicit pressure interrupts local tuning and creates architecture-
  family transitions.
- Countermechanism: forced novelty arrives at unproductive times, breaks strong
  lineages, or produces cosmetic compliance.

### RQ3 — interaction

Does portfolio memory amplify or suppress the transition intervention?

- **H3 (directional exploratory):** `(C3-C2) - (C1-C0) > 0` if alternative
  branches give assumption-changing proposals useful comparative evidence.
- A negative interaction is plausible if portfolio memory already supplies
  diversity or makes transition prompts cognitively overloaded.

### RQ4 — no-search reference

How many mechanisms arise from equal-budget independent proposals without
adaptive feedback? N0 is descriptive, not a fifth factorial cell. Compare its
distribution and efficiency with cells, but do not include it in H1–H3.

## 4. What counts as a Layer-B-qualified mechanism

A packet is already Layer-A-valid. Layer B asks whether the parent-to-candidate
delta instantiates a coherent, testable mechanism rather than merely moving a
knob.

Qualify (`1`) only when all are true:

1. The source delta implements the claimed mechanism, not only the final
   message.
2. The change alters a computation, information path, representation,
   optimization mechanism, data curriculum mechanism, or systems mechanism in
   a way that could causally affect the task.
3. The hypothesis makes a falsifiable directional or comparative claim.
4. The delta is attributable enough to name one primary mechanism.

Do not qualify (`0`) when the delta is only:

- width, depth, head count, batch size, learning-rate, schedule scalar, seed, or
  threshold tuning with no new mechanism;
- removal/pruning of an existing component without a mechanistic replacement;
- a bug fix, syntax fix, logging change, evaluator manipulation, or dead code;
- renaming/reformatting/refactoring with equivalent computation;
- a large bundle whose effect cannot be attributed to a coherent hypothesis;
- claimed novelty not present in the code.

Parameter changes can accompany a mechanism; they cannot be the mechanism.

### Cluster rule

Assign the same stable cluster label when two packets implement the same causal
idea even if dimensions, syntax, or surrounding code differ. Split clusters
when the information flow or learning mechanism differs in a way that would
support a separate ablation. Cluster labels must be global across all packets
in a task × framework campaign, not restarted per run.

Suggested label format:

```text
task-area__mechanism-family__short-name
```

Examples:

```text
adder__representation__carry-state-factorization
adder__attention__digit-aligned-routing
lm__optimizer__orthogonalized-matrix-update
lm__attention__alternating-local-global-window
lm__residual__learned-input-skip-gating
```

### AdderBoard examples

Likely mechanism families include arithmetic/carry representation, digit
alignment or position coding, causal attention routing, recurrent/shared block
structure, weight tying/factorization, modular arithmetic features, algorithmic
data curriculum, or a new training objective. Merely reducing embedding width,
removing one layer, pruning individual scalars, or extending training is not a
new mechanism.

### nanoGPT examples

Likely families include attention topology, state-space/recurrent mixing,
residual/value pathways, normalization, activation/MLP computation, embedding
sharing, optimizer update geometry, curriculum/data ordering, precision/kernel
strategy, or compute allocation. A different depth, batch size, LR, warmdown
ratio, or window length alone is tuning.

## 5. Reviewer protocol

1. Freeze this rubric before packet export.
2. Use two reviewers who cannot see condition/run/opportunity, Layer A score,
   event order, or the private mapping.
3. Review in the randomized `packet_order.tsv` order.
4. Compare `parent/` and `candidate/`; read hypothesis/edit only as a claim to
   verify against source.
5. Record qualification, cluster, primary mechanism, reviewer ID, and concise
   evidence in notes.
6. Before adjudication, calculate qualification agreement (percent plus an
   appropriate chance-corrected statistic) and cluster agreement on jointly
   qualified packets. Cluster agreement may require pairwise co-clustering or
   adjusted Rand index rather than exact arbitrary label names.
7. Adjudicate while still blinded. Freeze the adjudicated TSV hash before
   opening `private/`.
8. Report how many packets were invalid before Layer B, qualified, and merged as
   repetitions.

Potential residual unblinding: transition-generated hypotheses may use language
such as “challenge an assumption,” and portfolio-derived code may reveal richer
ancestry. Report this limitation and ask reviewers after adjudication to guess
condition; summarize guess accuracy as a blinding check.

## 6. Primary and secondary outcomes

### Primary

- Distinct Layer-B-qualified mechanism clusters per complete run.

### Prespecified process/secondary outcomes

- Layer-A-valid proposal count and rate.
- Layer-B-qualified packet count and rate.
- Novel clusters per proposal, per evaluator call, per million tokens, and per
  evaluator-hour.
- Time/proposal index of first new cluster and last new cluster.
- Cumulative unique-cluster curve.
- Final Layer A objective and Layer C result.
- Retention rate, strict-improvement rate, and invalid/failure categories.
- Parent-selection entropy and concentration by lineage.
- Portfolio occupancy and lineage survival time.
- Transition-checkpoint compliance and immediate/lagged cluster yield.
- Input, cached input, output, reasoning output, evaluator calls, wall compute,
  and termination reason.

Do not redefine a secondary efficiency metric as primary after seeing results.

## 7. Statistical analysis notes

### Unit and contrasts

Runs—not proposals, packets, or tokens—are independent experimental units.
Report cell means and the exact three contrasts emitted by `analysis.py`, plus
within-block contrasts. Analyze each task × framework stratum separately.

### Uncertainty

Three blocks per cell matches common autonomous-research benchmark practice but
is small. Show all run points. Prefer effect sizes and intervals. If using a
count regression, specify it before unblinding, include block effects, check
overdispersion/zero inflation, and do not imply asymptotic certainty from packet
counts. A block bootstrap has only three resampling units and should be labeled
descriptive/sensitivity analysis.

### Cross-stratum synthesis

The four task × framework strata are not exchangeable by default. A defensible
synthesis is:

1. show all stratum effects;
2. assess sign/scale consistency;
3. optionally fit a hierarchical model with task/framework variation;
4. keep the raw-stratum conclusions primary under the present sample size.

### Missingness and failures

Invalid proposals contribute no Layer B packet but remain part of the run
budget. Report invalid rates by condition: a treatment that yields “novel” ideas
but many malformed candidates has a meaningful cost. Do not exclude low-
performing complete runs. An incomplete campaign is not silently imputed.

### Multiplicity

H1–H3 are the three prespecified contrasts for one stratum. Four strata create
replicated/synthesized tests, not permission to report only favorable strata.
Label all mechanism-family subgroup analyses and transition timing analyses
secondary unless preregistered before unblinding.

## 8. Figures and tables worth preparing

### Main paper

1. **2×2 design diagram:** memory on one axis, transition policy on the other,
   with identical controls surrounding all cells.
2. **Cell plot:** run-level cluster counts, cell means, paired block lines, and
   N0 in a visually separated panel.
3. **Contrast forest plot:** memory, transition, and interaction for each task ×
   framework stratum.
4. **Cumulative discovery curves:** unique qualified clusters versus proposal
   opportunity, with checkpoint markers at 20/40/60/80.
5. **Efficiency plot:** clusters per million tokens and evaluator-hour, retaining
   raw run points.
6. **Mechanism composition:** heatmap of cluster families by condition; counts
   should not be read as independent tests.

### Appendix

- Lineage trees with retained/rejected nodes and selected-parent edges.
- Portfolio occupancy and parent-selection concentration over time.
- Tokens, evaluator calls, and compute by run.
- Failure-category stacked bars.
- Layer A final score versus mechanism diversity scatterplot.
- Reviewer agreement and adjudication flow.
- Transition event-study plots: new clusters in prespecified windows before and
  after checkpoints, clearly secondary.
- N0 independent proposal distribution.

Avoid bar charts without raw points and avoid cumulative-proposal curves that
pretend within-run points are independent replicates.

## 9. Threats to validity

### Internal validity

- Codex provider sampling is not seedable and model aliases may evolve. Freeze
  model name/settings, execute blocks close in time, record CLI/provider dates,
  and use blocked round-robin order.
- Prompts diverge after trajectories diverge because visible candidates and
  metrics are treatment-mediated state. This is intended; only the generating
  template and exogenous treatment slots are held fixed.
- Portfolio prompts are longer. Additional context/token cost is part of the
  memory treatment, so report both raw discovery and token-normalized efficiency.
- The transition instruction is encouragement, not guaranteed compliance.
  Analyze blinded realized mechanisms, not proposal labels alone.
- Reviewer judgments are subjective. Use two blinded reviewers, parent deltas,
  a frozen rubric, agreement, and adjudication.
- A scientific runtime mismatch fails closed, but package/driver/provider
  behavior still requires environment receipts.

### Construct validity

- A mechanism cluster is a human-coded proxy for research diversity, not proof
  of novelty to the scientific literature.
- Counting clusters weights a one-off weak mechanism and a repeatedly successful
  mechanism equally. Keep performance and replication as secondary dimensions.
- The controller’s portfolio is `K=4` with one specific selection/retention rule;
  conclusions do not cover every population method.
- The OpenEvolve stratum is a controlled proposal adapter, not native OpenEvolve
  end-to-end.

### External validity

- AdderBoard is small and synthetic; nanoGPT is compute-heavy and language-
  modeling-specific.
- Both generators are Codex with one model/configuration. Results need not
  generalize to other model families, human researchers, or longer horizons.
- nanoGPT’s fixed-time score is H100/software-image-specific.
- Layer C for nanoGPT is replication on the pinned validation procedure, not a
  new distribution.

### Conclusion validity

- Three blocks give noisy count estimates and unstable interactions.
- Failed or prematurely terminated runs can be condition-dependent.
- Multiple tasks, frameworks, secondary metrics, and mechanism families create
  many analytical degrees of freedom. Publish the prespecified outcomes and all
  strata.

## 10. Data provenance map

| Claim/evidence | Authoritative artifact |
|---|---|
| Frozen treatment and budget | `inputs/protocol.json`, `campaign.json` hashes |
| Assignment/order/seed | `schedule.json`, run `manifest.json` |
| Starting artifact | baseline/support/candidate hashes |
| Prompt actually sent | `opportunities/NNNN/prompt.md` and manifest hash |
| Codex trace and usage | `opportunities/NNNN/codex/*.jsonl` |
| Candidate and parent | content-addressed `candidates/`, event parent IDs |
| Evaluator result | evaluation JSON/stdout/stderr and event record |
| Retention/portfolio | `events.jsonl`, atomic `state.json` |
| Primary annotations | hashed adjudicated Layer B TSV |
| Treatment mapping | sealed `private/mapping.json` after adjudication |
| Contrasts | `scored/factorial_estimates.json` plus analysis commit |
| Final replication/holdout | `sealed-layer-c/summary.json` and workspaces |

Use event logs as the source of truth. Narrative Codex messages and filenames
are supporting context, not the scientific record.

## 11. Suggested paper structure

1. **Introduction:** autonomous research often conflates memory, search policy,
   and infrastructure; final scores obscure what was discovered.
2. **Related work:** FML-bench, Heuresis, Autoresearch, OpenEvolve, trace-based
   evaluation, and long-horizon architecture research.
3. **Method:** 2×2 design, N0, controller, tasks/frameworks, blocking, budgets,
   and Layer A/B/C separation.
4. **Mechanism annotation:** blinded packets, rubric, reviewers, agreement.
5. **Results:** primary contrasts by stratum, N0, efficiency, realized mechanism
   families, final task performance.
6. **Process analysis:** lineages, transition timing, failure modes, token/compute
   tradeoffs.
7. **Discussion:** when diversity interventions help, what they cost, and why
   outcome and mechanism diversity differ.
8. **Limitations/ethics:** provider drift, reviewer subjectivity, task breadth,
   compute, reward hacking, and reproducibility.
9. **Reproducibility appendix:** source/config hashes, commands, environments,
   complete runs, deviations, and artifact schema.

## 12. Results table templates

### Run-level primary table

| Task | Framework | Block | Condition | Clusters | Valid proposals | Tokens | Eval hours | Final Layer A | Layer C | Termination |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|

### Contrast table

| Task | Framework | C0 | C1 | C2 | C3 | Memory effect | Transition effect | Interaction |
|---|---|---:|---:|---:|---:|---:|---:|---:|

### Reviewer table

| Task/framework | Packets | Reviewer A qualified | Reviewer B qualified | Qualification agreement | Cluster agreement | Adjudicated clusters |
|---|---:|---:|---:|---:|---:|---:|

## 13. Protocol deviation log template

Append one row before analysis whenever anything differs from paper v1:

| Timestamp UTC | Campaign/run IDs | Before/after first affected opportunity | Frozen item | Reason | Exact change | Data disposition | Approved by |
|---|---|---|---|---|---|---|---|
| 2026-08-20 | Prospective `c0c3-workshop-pilot-parallel-v1` campaigns | Before campaign creation and before every affected opportunity | Protocol version, execution rule, per-run opportunity budget, and transition schedule | Deadline-bounded workshop evidence collection with concurrent condition calls | Added protocol 1.1: three blocks, 30 opportunities, checkpoints 10/20, `blocked_parallel_condition_rounds_v1`; N0 remains serialized after each C0–C3 group | Analyze as its own protocol stratum; never pool with `paper_v1`; disclose partial recovery subsets | Operator |

Never overwrite an earlier row. If code/config changed, record old/new hashes and
create a new campaign/protocol identifier.

## 14. Submission-time evidence checklist

- [ ] All four task × framework campaigns use the intended protocol version.
- [ ] All launch validations passed before first opportunity.
- [ ] Source, dependency, Codex, hardware, data, and calibration receipts exist.
- [ ] Every expected run completed or is explicitly disclosed as incomplete.
- [ ] No direct diagnostic run bypassed frozen ordering for included data.
- [ ] Layer B was unavailable until all runs completed.
- [ ] Two independent blinded reviews and agreement statistics are archived.
- [ ] Adjudicated annotations were hashed before condition mapping opened.
- [ ] Primary results include every complete run and all prespecified strata.
- [ ] N0 is visually and analytically separate from the factorial.
- [ ] Invalid/failure rates and full compute/token costs are reported.
- [ ] nanoGPT Layer C is called replication, not holdout/generalization.
- [ ] Controlled OpenEvolve is not mislabeled as native end-to-end OpenEvolve.
- [ ] Limitations include provider nondeterminism, prompt-length cost, annotation
      subjectivity, small block count, and task/framework scope.
- [ ] Artifact archive hashes and public/private release plan are complete.
