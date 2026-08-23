# Challenging the Search

## Experimental Plan for Studying Memory, Assumption Challenges, and Autonomous Research Dynamics

**Document status:** Shareable research plan
**Date:** August 21, 2026
**Provisional venue:** NeurIPS workshop
**Primary systems:** AutoResearch-style sequential search and OpenEvolve-style population search

## Short summary

This project studies research agents as experimental subjects. It asks how two controlled changes affect the way agents conduct research:

1. **Portfolio memory:** showing an agent a compact record of several past approaches, including successes, failures, and abandoned alternatives.
2. **Assumption challenge:** asking an agent to identify a premise behind its current direction, construct an alternative explanation, and choose an experiment that can distinguish the two.

The main outcomes describe the research process rather than the final benchmark score. We measure which experiments agents choose, whether they seek confirmation or falsification, how they react to contradictory evidence, how long alternative ideas survive, and whether research populations converge around shared assumptions.

The project contains five experiments:

| Experiment | Purpose | Main comparison |
|---|---|---|
| E0. Retrospective baseline | Describe normal research dynamics and build the annotation codebook | Existing AutoResearch and OpenEvolve trajectories |
| E1. Manipulation pilot | Check whether the challenge changes implemented experiments rather than prose alone | Neutral prompt versus challenge prompt |
| E2. Checkpoint-fork study | Estimate the immediate causal effect from matched research states | Four branches cloned from the same checkpoint |
| E3. Full-trajectory study | Measure cumulative effects, fixation, convergence, and reversion | Scheduled interventions across complete runs |
| E4. Cross-system and cross-task replication | Test whether the observed dynamics transfer | AutoResearch versus OpenEvolve, then a second task |

Final performance, validity, and cost remain downstream outcomes. A useful result may show that an intervention changes research reasoning without improving the final score. Such a result would identify a gap between producing dissent and developing it into successful research.

## 1. Research objective

### Central question

> How do portfolio memory and explicit challenges to current assumptions change the reasoning, experiment selection, evidence interpretation, and trajectory dynamics of autonomous research agents?

### Object of study

The project studies two related aspects of autonomous research:

- **Research process:** the observable sequence of proposals, experiments, selections, branches, reversions, and abandoned directions.
- **Research logic:** the hypotheses agents use, the assumptions embedded in those hypotheses, the evidence they cite, and the way they interpret new results.

The causal chain is:

\[
\text{intervention}
\rightarrow
\text{research reasoning}
\rightarrow
\text{experiment choice}
\rightarrow
\text{interpretation}
\rightarrow
\text{future search}
\rightarrow
\text{eventual outcome}.
\]

The paper concentrates on the middle of this chain. Benchmark performance appears later as a consequence of research behavior.

### Intended contribution

The paper should contribute:

1. A causal study of how assumption challenges change research decisions.
2. A process-level account of how portfolio memory changes evidence use and path dependence.
3. A trajectory dataset linking stated explanations, implemented experiments, results, interpretations, and descendants.
4. Evidence about whether intervention effects persist, disappear, or get removed by the search system.

## 2. Research questions

### RQ1. Formation of research commitments

- How fast does an agent settle on one explanation or mechanism family?
- How much evidence supports that commitment?
- Does the agent distinguish an observed improvement from a causal explanation?
- Do several lineages inherit the same premise despite code-level diversity?

### RQ2. Immediate response to a challenge

- Which assumption does the agent choose to question?
- Does it challenge a core mechanism or a minor implementation detail?
- Does it form a competing explanation?
- Does it choose an experiment whose outcomes favor different explanations?

### RQ3. Interpretation of contradictory evidence

- Does the agent weaken or reject the implicated hypothesis?
- Does it narrow the claim, request replication, or blame implementation noise?
- Does it change the criterion for success after seeing the result?
- Does it rewrite its earlier explanation to preserve the research direction?

### RQ4. Downstream trajectory

- Does the challenged direction survive for several research decisions?
- Does the agent return to its pre-intervention logic?
- Does it combine the alternative with an older approach?
- Does the population converge around a revised shared assumption?

### RQ5. Role of portfolio memory

- Does memory help the agent use failures and minority ideas?
- Does it reduce repeated experiments?
- Does it anchor the agent on approaches represented in the visible portfolio?
- Does it make challenges more informed or more conservative?

### RQ6. Search-topology moderation

- Does sequential search revert to the incumbent more often?
- Does population search preserve challenged ideas for longer?
- Can a diverse program population still share one dominant research logic?

## 3. Experimental factors

### Factor M: visible research memory

The treatment controls the history visible to the proposing language model. It does not change the framework's internal selection rules.

#### M0: sequential memory

The agent sees:

- Its current candidate or selected parent.
- The most recent experiment.
- The latest public feedback.
- Its current research note.

#### M1: portfolio memory

The agent also sees a compact sample of past directions:

- One successful approach.
- One valid failure.
- One distant alternative.
- One abandoned direction.

Each portfolio entry contains:

```text
Approach
Proposed mechanism
Experiment performed
Observed result
Agent interpretation
Reason retained or abandoned
```

The retrieval rule, number of entries, ordering, and token allowance must remain fixed across the study. The paper should describe the treatment as one operational form of portfolio memory.

OpenEvolve keeps its internal program database in both memory conditions. M0 hides portfolio summaries from the LLM while leaving the evolutionary system intact. AutoResearch receives the same M0 or M1 visible context.

### Factor I: scheduled deliberation prompt

At fixed research opportunities, the system injects one of two prompts.

#### I0: neutral evidence review

> Review the available results. Choose the next experiment and explain which evidence supports that choice. State what result you expect.

#### I1: assumption challenge

> Identify one claim that your current research direction treats as true. Select a claim whose rejection would change your next decision. State an alternative explanation, describe evidence that would favor each explanation, and choose the next experiment based on that distinction.

Both conditions receive the same model, context size, tools, execution budget, timing, and total token allowance. The intervention must not change rewards, selection, branch protection, or acceptance rules.

### Factor F: research framework

- **F0:** AutoResearch-style sequential search.
- **F1:** OpenEvolve-style population search.

The framework serves as a moderator. The study estimates intervention effects within each system and does not treat the raw framework difference as a clean causal comparison.

## 4. Standard research record

All conditions produce the same short lab note before and after each experiment. This record captures public research commitments without requesting private chain-of-thought.

### Before the experiment

```text
Current explanation:
Evidence relied upon:
Next experiment:
Expected result:
Decision that each possible result would support:
```

### After the experiment

```text
Observed result:
Interpretation:
Does this change the current explanation?
Next decision:
```

The intervention prompt precedes this common record. Annotators compare the stated purpose with the implemented experiment and subsequent decision.

## 5. Units of analysis

### Decision event

One research decision is:

\[
D_t=(S_t,R_t,A_t,O_t,U_t),
\]

where:

- \(S_t\): research state before the decision.
- \(R_t\): stated rationale.
- \(A_t\): chosen action or experiment.
- \(O_t\): observed result.
- \(U_t\): interpretation and subsequent update.

### Intervention episode

An episode contains the scheduled prompt and the following \(h\) decisions:

\[
E_t=\{D_t,D_{t+1},\ldots,D_{t+h}\}.
\]

Use a short horizon such as \(h=3\) in the checkpoint-fork study.

### Complete trajectory

A trajectory contains one full research run. Use complete runs as the independent units for cumulative effects. Decisions within one run remain dependent observations.

## 6. Research-logic codebook

Develop this codebook from untreated baseline trajectories, calibrate annotators, then freeze it before the main intervention study.

### A. Research move

Label the implemented experiment:

- Local refinement.
- Hyperparameter search.
- Mechanism modification.
- Alternative mechanism.
- Ablation.
- Counterexample or boundary test.
- Confound test.
- Replication.
- Reversion.
- Recombination.
- Evaluator exploitation.
- Unresolved.

### B. Epistemic purpose

Label the stated purpose:

- Improve performance.
- Confirm a hypothesis.
- Falsify a hypothesis.
- Distinguish competing explanations.
- Diagnose a failure.
- Estimate robustness.
- Measure a boundary.
- Reproduce a result.
- Explore without a stated hypothesis.

Research move and epistemic purpose require separate labels. An ablation may serve optimization rather than falsification.

### C. Assumption level

- Task structure.
- Architecture family.
- Computational mechanism.
- Optimization.
- Capacity.
- Data or curriculum.
- Evaluation validity.
- Generalization.
- Implementation correctness.
- Resource constraint.

### D. Evidence source

- Latest result.
- Several results from one lineage.
- Cross-lineage comparison.
- Retrieved failure.
- Retrieved successful alternative.
- Generic model prior.
- External literature or documentation.
- No identifiable evidence.

### E. Response to evidence

- Retains hypothesis.
- Weakens hypothesis.
- Rejects hypothesis.
- Narrows scope.
- Adds an auxiliary explanation.
- Requests replication.
- Attributes the result to implementation failure.
- Attributes the result to noise.
- Changes the hypothesis without acknowledging conflict.
- Provides no belief update.

### F. Research displacement

- **D0:** Same hypothesis and mechanism.
- **D1:** Implementation-level change.
- **D2:** Different test of the same mechanism.
- **D3:** Competing explanation within the same family.
- **D4:** Different mechanism or architecture family.
- **D5:** Different formulation of the research problem.

## 7. Primary outcomes

### P1. Discriminating experiment rate

\[
Y^{\mathrm{disc}}_t=
\mathbf{1}[\text{the experiment gives competing explanations different predicted observations}].
\]

This outcome tests whether the intervention changes scientific test selection.

### P2. Assumption displacement

\[
Y^{\mathrm{disp}}_{t,h}=
\mathbf{1}[\text{the research logic reaches D3, D4, or D5 within horizon }h].
\]

Use \(h=3\) for the fork study.

### P3. Evidence-responsive revision

Among episodes that produce evidence inconsistent with the agent's prediction:

\[
Y^{\mathrm{rev}}=
\mathbf{1}[\text{the agent weakens, narrows, rejects, or retests the implicated hypothesis}].
\]

This outcome separates challenge generation from belief revision.

## 8. Secondary process outcomes

### Research-move distribution

Estimate changes in refinement, ablation, alternative mechanisms, replication, reversion, and boundary tests.

### Logic entropy

For research-logic category proportions \(p_k\):

\[
H_{\mathrm{logic}}=-\sum_k p_k\log p_k.
\]

Report the category distribution with entropy because high entropy can reflect broad exploration or incoherent switching.

### Logic persistence

\[
P(L_{t+1}=L_t).
\]

### Hypothesis half-life

Count the decisions until the agent rejects, replaces, narrows, or stops using a stated hypothesis.

### Contradictory-evidence tolerance

Count inconsistent observations before the agent revises the implicated hypothesis.

### Rationale-action alignment

Label the relation between the stated purpose and implemented change:

- Aligned.
- Partially aligned.
- Unrelated.
- Contradictory.
- Unclear.

### Interpretation stability

Measure whether the agent follows its pre-experiment decision rule or changes the criterion after seeing the result.

### Reversion

\[
R_h=\mathbf{1}[\text{the agent returns to its pre-intervention logic within }h\text{ decisions}].
\]

### Population and lineage dynamics

- Active research directions.
- Branch creation and survival.
- Dominant research-logic share.
- Dominant ancestor share.
- Architecture-region revisits.
- Cross-branch recombination.
- Re-entry after abandonment.

### Memory use

- References to retrieved evidence.
- Reuse of a displayed mechanism.
- Avoidance of a displayed failure.
- Recombination of portfolio entries.
- Proposals outside the displayed portfolio.

## 9. Downstream outcomes

Analyze these after the process outcomes:

- Best task score.
- Area under the score curve.
- Valid-candidate rate.
- Final incumbent quality.
- Held-out robustness.
- Cost and token use.
- Number of distinct high-performing candidates.

These outcomes test whether process changes have practical consequences. They do not determine whether the intervention changed research behavior.

## 10. Hypotheses

### H1. Challenges change experimental purpose

Assumption challenges increase falsifying and discriminating experiments relative to neutral evidence review.

### H2. Challenges increase research displacement

Challenge episodes produce larger changes in research logic and mechanism family over the following three decisions.

### H3. Challenges improve response to contradiction

Agents revise or retest hypotheses more often after contradictory evidence in challenge conditions.

### H4. Portfolio memory changes challenge content

Portfolio memory causes agents to target assumptions shared across prior directions and use cross-lineage evidence.

### H5. Portfolio memory can anchor reasoning

Portfolio conditions increase reuse and recombination of displayed approaches while reducing proposals outside the displayed set.

### H6. Search topology moderates persistence

Sequential search returns to the incumbent logic more often, while population search preserves challenge-originated directions for longer.

### H7. Process effects can exceed outcome effects

The treatment's effect on discriminating experiments and assumption displacement may exceed its effect on final performance.

## 11. Experiment E0: retrospective baseline

### Purpose

Describe untreated research behavior and develop the annotation codebook.

### Data

Use complete existing trajectories from AutoResearch and OpenEvolve, including rejected candidates and ordinary agent messages where available.

### Analyses

- Research-move frequency.
- Logic transition matrices.
- Time to dominant research logic.
- Evidence-source distribution.
- Response to positive and negative feedback.
- Hypothesis half-life.
- Architecture diversity versus logic diversity.

### Deliverables

- Frozen codebook.
- Annotator handbook.
- Baseline transition matrices.
- Checkpoint-sampling rules.
- Variance estimates for later power calculations.

## 12. Experiment E1: manipulation pilot

### Purpose

Test whether the intervention changes implemented research behavior rather than generated prose.

### Design

Sample saved checkpoints and create the four memory-by-prompt branches. Run each branch for one to three decisions.

Suggested pilot:

- 8 checkpoints from AutoResearch.
- 8 checkpoints from OpenEvolve.
- 4 branches per checkpoint.
- 3 decisions per branch.

This yields 192 decision events nested within 16 matched checkpoints.

### Manipulation checks

- Specific assumption identified.
- Decision-relevant assumption selected.
- Competing explanation stated.
- Different predictions stated.
- Implemented action tests the stated assumption.
- Control condition contamination.

### Go/no-go rule

Proceed only if the challenge changes implemented experiment types and research displacement. Revise the prompt once if the arms differ in prose but not action. Exclude pilot data after a prompt revision.

## 13. Experiment E2: checkpoint-fork causal study

### Purpose

Estimate the local causal effect from matched research states.

### Design

Clone each checkpoint into:

| Branch | Memory | Prompt |
|---|---|---|
| F0 | Sequential | Neutral |
| F1 | Sequential | Challenge |
| F2 | Portfolio | Neutral |
| F3 | Portfolio | Challenge |

Run each branch for three to five decisions.

### Checkpoint strata

- Early search.
- Recent improvement.
- Plateau.
- Recent failed experiment.
- High lineage concentration.
- High population diversity.

Freeze checkpoint-state definitions before treatment assignment. Sample checkpoints without using their future intervention responses.

### Main outcomes

- Discriminating experiment rate.
- Assumption displacement.
- Evidence-responsive revision.
- Rationale-action alignment.
- Reversion within three decisions.
- Branch survival.

### Sample size

Use pilot-based power analysis. A provisional range is 25 to 40 independent checkpoints per framework. Cluster uncertainty by source trajectory when several checkpoints come from one run.

## 14. Experiment E3: full-trajectory study

### Purpose

Measure cumulative, persistent, and path-dependent effects.

### Design

Run complete trajectories under:

| Condition | Visible memory | Scheduled prompt |
|---|---|---|
| C0 | Sequential | Neutral |
| C1 | Sequential | Challenge |
| C2 | Portfolio | Neutral |
| C3 | Portfolio | Challenge |

Apply scheduled prompts at fixed opportunities, such as decisions 5, 10, 15, and 20. Use the same schedule across conditions.

### Main outcomes

- Logic concentration over time.
- New research-move rate.
- Hypothesis half-life.
- Contradictory-evidence tolerance.
- Branch survival and reversion.
- Memory anchoring.
- Architecture diversity versus logic diversity.

### Run count

Use complete runs as independent units. A provisional range is 8 to 16 runs per framework-condition cell. Freeze the final count through simulation-based power or precision analysis based on the pilot.

## 15. Experiment E4: cross-system and cross-task replication

### Cross-system analysis

Estimate memory and challenge effects within AutoResearch and OpenEvolve. Analyze framework-by-treatment interactions as moderation.

### Cross-task analysis

Add one task with:

- Several plausible mechanisms.
- Fast experiments.
- Interpretable failures.
- A scalar evaluator.
- Behavioral probes.

Candidate domains include symbolic regression, signal processing, or small-scale RL algorithm design.

The second task tests whether the observed dynamics reflect one benchmark or a broader property of research-agent behavior.

## 16. Statistical analysis

### Checkpoint-fork model

For checkpoint \(j\), memory condition \(M\), and prompt condition \(I\):

\[
Y_{jMI}=\alpha_j+\beta_M M+\beta_I I+\beta_{MI}MI+\epsilon_{jMI}.
\]

The checkpoint effect \(\alpha_j\) controls for all history before the fork.

Use conditional logistic or mixed-effects logistic models for binary outcomes. Use ordinal models or paired rank-based estimates for research displacement.

### Full-trajectory model

For process summary \(G_r\):

\[
G_r=
\beta_0+
\beta_M M_r+
\beta_I I_r+
\beta_{MI}M_rI_r+
\beta_F F_r+
\beta_{IF}I_rF_r+
\beta_{MF}M_rF_r+
\alpha_{\mathrm{block}(r)}+
\epsilon_r.
\]

Report marginal differences, confidence intervals, and block-aware randomization inference where possible.

### Event study

Center decision time on each scheduled intervention and analyze from three decisions before to five decisions after it. Plot:

- Research displacement.
- Mechanism distance.
- Epistemic purpose.
- Evidence-source use.
- Logic persistence.
- Branch creation and reversion.

### Sequence analysis

Estimate transitions between research moves:

\[
P_{ab}=P(L_{t+1}=b\mid L_t=a).
\]

Examples include refinement to ablation, failure diagnosis to replication, and alternative mechanism to reversion.

Do not count decision events from one trajectory as independent runs.

## 17. Qualitative process tracing

### Case selection

Use outcome-independent rules:

- Random challenge and control forks.
- Median-displacement episodes.
- Episodes with contradictory evidence.
- Episodes where rationale and action disagree.
- Long-lived and short-lived logic changes.

### Reconstruction template

For each case, document:

1. Pre-intervention research commitment.
2. Evidence supporting it.
3. Assumption challenged or preserved.
4. Alternative explanation.
5. Chosen experiment.
6. Observed result.
7. Agent interpretation.
8. Next three decisions.
9. Population or lineage consequence.
10. Final state of the implicated assumption.

Matched checkpoint forks provide strong qualitative comparisons because each branch shares the same preceding history.

## 18. Annotation protocol

### Development

1. Two annotators inspect 50 to 100 untreated decision events.
2. They draft operational labels and edge-case rules.
3. They label a separate calibration sample.
4. They resolve disagreements and freeze the codebook.
5. The main study excludes calibration events.

### Blinding

Hide:

- Condition name.
- Prompt wording.
- Framework where feasible.
- Final benchmark score.
- Future descendant success.

Annotators receive the local history needed to interpret the decision.

### Reliability

Report raw agreement, the category confusion matrix, Cohen's kappa or Krippendorff's alpha, and results before adjudication. LLM-generated labels may support triage but should not define confirmatory outcomes.

## 19. Validity threats

### Intervention contamination

The neutral prompt may induce reflection, and the challenge prompt may produce stylistic skepticism. Measure spontaneous challenges in the control arm and rationale-action alignment in both arms.

### Public rationale validity

The lab note records an external research commitment. It does not reveal the model's internal reasoning. Frame conclusions around expressed logic and subsequent behavior.

### Carryover

An intervention changes later research states. Use checkpoint forks for local causal effects and full runs for cumulative effects. Do not interpret later event-level comparisons as independent treatments.

### Framework confounding

AutoResearch and OpenEvolve differ in selection, memory, and population structure. Estimate treatment effects within frameworks.

### Task dependence

One architecture task may reward certain research moves. Add a second task or narrow the claim.

### Annotation subjectivity

Freeze operational definitions, blind annotators, report agreement, and retain an unresolved label.

### Provider drift

Record exact model identifiers, request parameters, timestamps, retry behavior, and block execution across a short provider window.

## 20. Interpretation of possible results

### Process and outcome both change

The intervention changes research reasoning and those changes correspond to different downstream outcomes.

### Process changes without outcome changes

The agent generates different experiments and revises assumptions, but the task score remains similar. This identifies a gap between epistemic exploration and productive development.

### Written logic changes without action changes

The agent complies in prose but implements the same kind of experiment. This limits claims about LLM reflection and motivates stronger action-level evaluation.

### Challenges create alternatives that disappear

The agent proposes dissent, but the framework's selection process removes it. This shows how research organization can dominate deliberation prompts.

### Portfolio memory broadens and anchors

Memory may reduce duplicate work while restricting proposals to displayed examples. A mixed effect would clarify the tradeoff between historical awareness and path dependence.

### Minimal intervention effect

A null result would show that lightweight assumption prompts do not alter autonomous research behavior under the tested systems and tasks.

## 21. Suggested execution order

### Week 1

1. Normalize existing trajectories.
2. Draft and calibrate the research-logic codebook.
3. Freeze checkpoint definitions.
4. Implement the two prompts and visible-memory contexts.
5. Run E1, the manipulation pilot.
6. Annotate action-level treatment fidelity.
7. Freeze the main checkpoint-fork protocol.

### Week 2

1. Run E2 across both frameworks.
2. Complete blinded annotation in batches.
3. Produce paired estimates and event-study plots.
4. Decide whether the intervention changes action and interpretation.

### Week 3

1. Run E3 complete trajectories.
2. Analyze fixation, persistence, convergence, and reversion.
3. Begin qualitative process tracing.

### Week 4

1. Run the second-task replication if resources permit.
2. Analyze downstream performance and cost.
3. Complete figures and draft the paper.

## 22. Core paper claim

The paper should aim to support a statement of this form:

> Scheduled assumption challenges changed which experiments research agents selected, how they interpreted conflicting evidence, and how long alternative research logics survived. Portfolio memory moderated these effects by supplying cross-lineage evidence while also anchoring agents on approaches represented in the visible archive.

The final benchmark result should qualify this claim rather than define it.

## 23. Closest related work

- [Failing to Falsify: Evaluating and Mitigating Confirmation Bias in Language Models](https://arxiv.org/abs/2604.02485)
- [Socratic Agents for Autonomous Scientific Discovery in High-Dimensional Physical Systems](https://arxiv.org/abs/2606.26722)
- [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://arxiv.org/abs/2502.09858)
- [Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty](https://arxiv.org/abs/2606.25198)
- [IDEAgent: Agentic Quality-Diversity Search for Research Idea Generation](https://arxiv.org/abs/2607.22375)
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://openreview.net/forum?id=IkmD3fKBPQ)

The differentiation is the causal, trajectory-level study of how interventions affect research logic, evidence response, persistence, and reversion across existing autonomous research systems.
