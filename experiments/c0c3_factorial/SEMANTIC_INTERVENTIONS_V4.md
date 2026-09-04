# Semantic intervention protocol v4

## Scientific question

Which semantic research interventions help an autonomous ML researcher, in
which parts of a trajectory, and through what observable changes to the
research process?

This is an exploratory multi-arm successor to the C0-C3 factorial. It uses the
unified v3 controller, task adapters, artifact model, evaluators, source
integrity checks, recovery rules, and append-only records. The scientific
outer design is v4 because intervention identity is no longer represented by
the four factorial cells.

The intervention is the research operation requested at a phase boundary, not
just the wording of one prompt. The analysis therefore treats prompt
compliance, implementation validity, mechanism evidence, retention, later
yield, generalization, and resource use as separate outcomes.

This design directly studies the AI system's effect on scientific practice.
Task performance supplies executable evidence, but it is not by itself the
paper's contribution. That framing fits AISCiK's emphasis on construct
validity, epistemic properties, reproducibility, reward/specification gaming,
homogenization, and the difference between task success and scientific
understanding.

## Experimental geometry

- 23 intervention arms.
- 3 independently seeded replicates per arm.
- 69 logical trajectories.
- 40 proposal opportunities per trajectory.
- One literal five-proposal prefix shared by all arms within a replicate.
- The first intervention occurs at proposal 6; later interventions occur at
  11, 16, 21, 26, 31, and 36.
- A Codex conversation persists for five proposals, then a new conversation
  begins from the bounded, verified evidence capsule and filesystem state.
- The same replicate seed, starting source, evaluator, model, reasoning effort,
  public evidence policy, proposal budget, and maximum evaluator budget are
  used across arms.
- The orchestrator issues every currently runnable subject call without an
  artificial Sol-worker ceiling by default. An operator may set a live
  campaign concurrent-run ceiling; it is polled every half-second and changes
  dispatch eligibility only, never the campaign's running/paused state. Local
  Fashion-MNIST evaluation is
  independently capped at six shared MPS slots, so research-call concurrency
  does not increase GPU training concurrency beyond that task-wide ceiling.
  Each campaign also has a live cooperative evaluator ceiling at or below six;
  the controller dashboard can change it without interrupting active training.
- Post-fork trajectories are scheduled independently. When one trajectory
  finishes an opportunity, its next opportunity is dispatched immediately;
  it does not wait for slower trajectories to reach a round boundary.
- The detached launcher preserves `--max-workers 0` as unbounded. The scheduler
  reloads its arm registry between dispatches, so a safely registered arm can
  begin while unrelated trajectories keep running; no wave barrier or
  supervisor restart is required.

The shared prefix requires only 15 physical proposal calls. Post-fork branches
require 2,415 calls, for 2,430 physical calls and 2,760 logical trajectory
records. Shadow-prefix records are explicitly marked so campaign cost is not
counted 23 times.

Three replicates are intentionally modest. This campaign prioritizes breadth
for exploratory treatment screening. Proposals are repeated observations
within a trajectory, not independent replicates. Results must show all three
replicates and effect sizes; they cannot turn 40 proposals into `n=40`.

## Intervention arms

| ID | Family | Research operation | Verbatim intervention prompt | How it works |
|---|---|---|---|---|
| `passive_control` | control | No extra phase-boundary direction | None — the intervention prompt is empty. | The intervention text is empty, so at each phase boundary the researcher receives the ordinary task contract, current verified evidence, and filesystem state without any added research strategy. This estimates what the same research system does when the controller does not direct its reasoning toward reflection, novelty, diagnosis, or another named operation. |
| `active_reflection_control` | control | Effort-matched generic evidence review | Before choosing the next change, review the available evidence carefully. Weigh the strongest positive and negative results, decide which next step is most justified, and implement the best coherent change you identify. | The researcher is asked to review the strongest positive and negative evidence, decide which next step is best justified, and implement one coherent change. Because it encourages comparable deliberation without naming a particular epistemic strategy, it helps separate the effect of extra reflective effort from the semantic content of the other interventions. |
| `assumption_challenge` | epistemic | Identify and test a load-bearing assumption | Before choosing the next change, identify a load-bearing assumption in the current designs and the strongest evidence for and against it. Test a genuinely different learned computation that challenges that assumption. Keep the implementation feasible, preserve supported components that are not needed for the test, and state what result would count against the alternative. | The researcher identifies an assumption on which the current designs depend, weighs the evidence for and against it, and implements a feasible learned computation that challenges it. Supported components not needed for the test are preserved, and the researcher must identify an outcome that would count against the proposed alternative, making the intervention open-ended but testable. |
| `restrictive_assumption_challenge` | prompt philosophy | Restore the original C0-C3 single-mechanism, mechanism-class-exclusion, and specific-summary philosophy | Challenge one load-bearing assumption behind the current candidate. Use the visible evidence and conversation history to choose an assumption whose failure would justify a different computational mechanism. Implement one coherent candidate from a genuinely different mechanism family, and make its evaluation discriminate between the old and alternative mechanisms.<br><br>This intervention must change how the task is represented or computed. A capacity change, optimizer or schedule change, scalar adjustment, pruning, parameter tying, deletion alone, or renamed variant of the same computation does not qualify. Do not repeat a mechanism family already tested in this run unless new evidence addresses a specific prior failure. Prefer the smallest implementation that cleanly tests the alternative. In the final summary, state the old assumption, the alternative mechanism, and the evaluation result that would support it. | This arm applies the historical C0-C3 wording: choose one load-bearing assumption, implement one coherent candidate from a genuinely different mechanism family, and make the evaluation discriminate between the old and new mechanisms. Capacity, optimizer, schedule, scalar, pruning, parameter-tying, deletion-only, and renamed-variant changes are disallowed; failed mechanism families require new evidence before reuse; and the summary must state the old assumption, alternative mechanism, and supporting result. |
| `alternative_mechanism` | epistemic | Implement a materially different learned computation | Step outside the current mechanism. Propose and implement a materially different learned way to represent or compute the task, chosen because the available results make it plausible and informative. Avoid merely retuning the present design. Make the alternative clean enough that its result reveals something about the mechanism. | The researcher must step outside the incumbent mechanism and build a materially different learned way to represent or compute the task. The alternative is selected using prior evidence, must be more than a retuning of the current design, and should be implemented cleanly enough that its result teaches something about the mechanism. |
| `failure_diagnosis` | epistemic | Diagnose causal failure before changing the design | Treat the recent failures as evidence rather than as isolated bad trials. Diagnose the most likely causal failure mode, distinguish it from plausible alternatives, and implement the next change that most directly tests or repairs that diagnosis. Do not repeat a failed form unless the diagnosis explains why the new version should differ. | Recent failures are treated as evidence from which the researcher infers a likely causal failure mode and distinguishes it from plausible alternatives. The next implementation is chosen to directly test or repair that diagnosis, and a previously failed form may be revisited only when the diagnosis explains why the new version should behave differently. |
| `evidence_contradiction` | epistemic | Center the result hardest to reconcile with the current account | Find the result that is hardest to reconcile with the current working explanation. Revise the explanation around that contradiction, then implement a change whose outcome discriminates between the old explanation and the revised one. Prefer a decisive, feasible test over a broad bundle of unrelated edits. | The researcher finds the observed result that most strongly conflicts with the current explanation and revises the explanation around that contradiction. It then implements a feasible, decisive change whose outcome can distinguish the old explanation from the revised one, rather than bundling several unrelated edits. |
| `counterfactual_design` | epistemic | Design as if the incumbent were unavailable | Imagine the current incumbent design were unavailable. Using only the task contract and the evidence, decide what learned system you would build instead. Implement the most informative feasible version of that counterfactual design, rather than recreating the incumbent under different names. | The researcher imagines that the incumbent cannot be used and independently asks what learned system it would build from the task contract and available evidence. It implements the most informative feasible version of that counterfactual, with an explicit warning not to recreate the incumbent under different names. |
| `falsification` | epistemic | Seek a discriminating result that could refute the current claim | Identify the strongest current mechanism claim and seek a serious falsification test. Implement a change or controlled contrast that would produce a different result if the claim is wrong. Choose a test that remains useful whether it succeeds or fails, and make the expected discriminating outcomes explicit. | The researcher selects the strongest current mechanism claim and designs a serious test that should produce a different result if the claim is wrong. Expected outcomes are stated explicitly, and the test is chosen to remain informative whether it supports or falsifies the claim. |
| `experimental_redesign` | methodological | Redesign the next experiment to isolate an uncertainty | Reconsider whether the recent sequence of trials is answering the right question efficiently. Redesign the next computational experiment so it isolates a consequential uncertainty with fewer confounds. Then implement that redesigned test within the available training and model constraints. | The researcher first asks whether the recent trial sequence is efficiently answering the right question, then redesigns the next computational experiment around one consequential uncertainty. The implementation aims to reduce confounds while remaining within the shared model and training constraints. |
| `causal_attribution` | methodological | Isolate the cause of a recent improvement or regression | Choose a recent improvement or regression whose cause is still ambiguous. Isolate the component most likely responsible and implement a change that makes the causal attribution cleaner. Favor interpretable evidence over another bundled improvement whose mechanism would remain unclear. | The researcher chooses a recent performance change whose cause is ambiguous, identifies the component most likely responsible, and implements a controlled change that isolates its contribution. The priority is interpretable causal evidence rather than another bundled modification that might improve performance without clarifying why. |
| `bottleneck_diagnosis` | strategic | Identify and attack the current limiting mechanism | Identify the bottleneck that currently limits further progress: representation, optimization, data use, capacity allocation, inference, or another concrete mechanism. Use the evidence to justify the bottleneck, then implement a focused change that attacks it without unnecessary collateral changes. | The researcher uses the evidence to locate the concrete factor limiting progress, such as representation, optimization, data use, capacity allocation, or inference. It then makes a focused change aimed at that bottleneck while avoiding collateral modifications that would obscure whether the diagnosis was correct. |
| `competing_hypotheses` | epistemic | Choose a test that separates competing explanations | Form at least two genuinely competing explanations for the current results. Choose and implement the next change whose possible outcomes best separate those explanations. Do not select a trial that both explanations predict equally well merely because it is easy. | The researcher formulates at least two genuinely competing explanations for the observed results and compares their predictions. It implements the next change whose possible outcomes best distinguish those explanations, rather than selecting an easy trial that both explanations predict equally well. |
| `information_gain` | methodological | Optimize the next trial for decision-relevant information | Choose the next change for expected information gain, not only for its chance of immediate improvement. Target an uncertainty whose resolution would materially change later design choices. Implement a feasible test with clearly different implications across its plausible outcomes. | The next change is selected for how much its result could alter later design choices, not solely for its immediate chance of improving the objective. The researcher targets a consequential uncertainty and implements a feasible test with clearly different implications across plausible outcomes. |
| `ablation_first` | methodological | Remove or control a component to test what is load-bearing | Use an ablation or controlled removal to determine which part of the current computation is actually load-bearing. Implement the most informative feasible ablation, keeping unrelated details fixed, and explain how each plausible result should change the next research direction. | The researcher removes or controls part of the current computation while keeping unrelated details fixed, choosing the feasible ablation that most directly reveals what is load-bearing. It also explains how each plausible result should redirect subsequent research, so a negative result remains useful. |
| `representation_reframe` | epistemic | Change the learned representation of the computation | Reconsider how the inputs, intermediate features, or outputs are represented. Identify a representation assumption that may be making the computation unnecessarily difficult, then implement a different learned representation that gives the model a meaningfully different computational route. | The researcher examines assumptions embedded in the input, intermediate-feature, or output representation and asks whether they make the task unnecessarily difficult. It then implements a different learned representation that gives the model a meaningfully different computational route rather than merely resizing the existing one. |
| `mechanism_recombination` | strategic | Cleanly combine complementary supported mechanisms | Look across the strongest distinct ideas in the available evidence. Combine compatible mechanisms only where their causal roles are complementary, not as an indiscriminate bundle. Implement a clean recombination whose result can show whether the mechanisms reinforce or interfere with one another. | The researcher surveys the strongest distinct ideas in the evidence and combines only mechanisms whose causal roles appear complementary. The recombination is kept clean enough for its result to show whether the mechanisms reinforce or interfere with one another, instead of functioning as an uninterpretable bundle. |
| `constraint_inversion` | strategic | Reverse a habitual priority to expose a neglected mechanism | Temporarily reverse one habitual design priority and ask what mechanism becomes attractive under the inverted priority. Use that perspective to expose a neglected computation, then implement a version that still satisfies the actual task contract and can be evaluated fairly. | The researcher temporarily reverses a habitual design priority and uses that altered perspective to identify a mechanism the normal priority may have hidden. It then implements a fair, evaluable version that still satisfies the actual task contract, making the inversion a tool for discovery rather than a change to the objective. |
| `exploitation_focus` | strategic control | Make the strongest evidence-led local refinement | Concentrate on the strongest supported line of work. Identify the smallest evidence-backed refinement most likely to improve the verified objective, preserve everything not implicated by the evidence, and implement that focused refinement. | The researcher stays on the best-supported line of work and chooses the smallest evidence-backed refinement most likely to improve the verified objective. Components not implicated by the evidence are preserved, producing a deliberate exploitation policy against which the more exploratory interventions can be compared. |
| `uncertainty_calibration` | epistemic | Test the consequential claim with greatest uncertainty | List the important claims you are least certain about and calibrate that uncertainty against the observed results. Select the uncertain claim with the greatest consequence for design, then implement a test that can substantially update confidence in it. | The researcher lists important claims it is least certain about, calibrates those uncertainties against observed results, and selects the uncertain claim with the greatest consequence for design. It implements a test intended to substantially update confidence in that claim rather than acting on unexamined confidence. |
| `assumption_plus_falsification` | limited combination | Challenge an assumption through a falsifiable alternative | Identify a load-bearing assumption and the strongest evidence for and against it. Replace it with a genuinely different learned mechanism, but design the implementation as a falsification test: state what result would undermine the alternative and keep unrelated components fixed enough to make that result interpretable. | This arm combines assumption replacement with a falsification-first test: the researcher identifies a load-bearing assumption, weighs evidence for and against it, and substitutes a genuinely different learned mechanism. Unrelated components are held fixed and the researcher states what result would undermine the alternative, preserving interpretability in both success and failure. |
| `diagnosis_plus_redesign` | limited combination | Diagnose failure and construct a discriminating redesign | Diagnose the causal pattern behind the recent failures, including at least one competing explanation. Then redesign the next computational experiment so its outcomes distinguish those explanations and directly test the proposed repair. Implement the smallest feasible version that preserves that discrimination. | The researcher diagnoses the causal pattern behind recent failures, includes at least one competing explanation, and redesigns the next experiment to distinguish those explanations while testing the proposed repair. It implements the smallest feasible version that preserves this discriminating structure. |
| `periodic_full_refresh` | memory control | Every five proposals, retain the incumbent model but restart search as if it were the starting design | None — the intervention prompt is empty; this arm is implemented through its phase-start state policy. | At each five-proposal boundary, the latest retained model becomes the new starting artifact while cumulative accounting and private audit records remain intact. The subject-visible outcome history, candidate population, developmental archive, parent history, and conversation are cleared, and a deterministic new search seed is used, isolating the effect of forgetting research history without discarding the model progress already achieved. |

The prompt text is frozen into each created campaign with file and tree hashes.
The labels above support analysis; the subject sees only the task, its available
design evidence, the normal research contract, and the relevant direction.

The open-ended `assumption_challenge` and
`restrictive_assumption_challenge` arms are an intentional prompt-philosophy
contrast. The latter reproduces the historical continuous C0-C3 intervention:
it requests one coherent alternative mechanism, excludes capacity, optimizer,
scalar, pruning, parameter-tying, deletion-only, and renamed-variant changes,
limits revisiting failed mechanism families without new evidence, and asks for
the old assumption, alternative mechanism, and discriminating result. The
former leaves the mechanism choice substantially more open.

## Shared-prefix causal comparison

Within replicate `r`, every arm inherits byte-identical candidate snapshots,
events, Layer A results, and verified evidence through proposal 5. Only three
prefix trajectories are physically executed. At proposal 6, each arm begins a
new five-proposal session and receives its registered intervention. This makes
the first post-fork phase a direct matched-prefix contrast.

Subsequent phases are trajectory-level effects rather than identical-state
effects because each arm has accumulated different evidence. Both estimands
are useful and must remain distinct:

1. immediate matched-prefix effect during proposals 6-10;
2. cumulative adaptive-trajectory effect through proposal 40.

## Five-proposal sessions

A fresh provider conversation starts at 1, 6, 11, 16, 21, 26, 31, and 36.
Within each five-proposal phase the researcher can follow through on a local
hypothesis. Across phases, raw conversational anchoring is removed while the
filesystem and deterministic evidence capsule preserve verified scientific
memory. This is neither every-proposal amnesia nor an unbounded chat.

Old session IDs remain in private append-only registries. Resetting a session
never deletes results or changes opportunity accounting.

The `periodic_full_refresh` arm applies a stronger reset at proposals 6, 11,
16, 21, 26, 31, and 36. It preserves cumulative budget accounting, verified
metrics, and the latest retained model artifact, then treats that artifact as
the new starting design. It clears the subject-visible outcome history,
candidate population, developmental archive, parent history, and conversation
session. Each phase receives a deterministic new search seed while the paired
replicate's evaluator/data seed remains unchanged. Private append-only events
and retired native-OpenEvolve checkpoints remain available for audit but are
not exposed to the subject.

The three refresh trajectories in the initially active campaign inherit their
replicate's already-completed proposals 1-5 with zero additional physical
prefix calls. Future campaigns include the arm at creation time.

## Multi-fidelity evaluation

### Fashion-MNIST

The default ladder presents checkpoints near 25,000, 50,000, then exactly
100,000 training examples. It is one uninterrupted training trajectory, not
three retrainings. Intermediate checks occur after the batch that crosses the
requested rung, so they do not split optimizer batches or change the training
sequence. Weak candidates may stop after a screening rung. A candidate cannot
replace the incumbent unless it reaches and passes the common 100,000-example
evaluation. The official 10,000-image test split remains sealed until final
evaluation.

The editable `train.py` owns two safe literal policies:

```python
EVALUATION_LADDER = [25_000, 50_000, 100_000]
EVALUATION_PROMOTION_THRESHOLDS = [0.82, 0.87, None]
```

The researcher may edit intermediate rungs and thresholds. The evaluator reads
them with `ast.literal_eval`; it never imports candidate code to obtain the
policy. It enforces 10,000-100,000 examples, at most six rungs, appends the
100,000-example terminal rung if omitted, and requires terminal confirmation.
An absent or malformed declaration falls back to the campaign default and is
recorded in the ladder receipt.

### AdderBoard

The default ladder is 5k, 10k, 15k, 20k, 25k, then 30k training steps. It stops
at the first rung reaching 99% accuracy and otherwise rejects after 30k. Each
rung is an evaluator-isolated deterministic training budget because the
current candidate CLI does not expose a trustworthy evaluator-owned resume
contract; this prevents a 5k failure from ending the proposal, at the cost of
repeating the shared prefix of the training trajectory. An agent may add or
edit a literal `EVALUATION_LADDER` in editable `src/train.py`. The evaluator
constrains it to 1k-30k, at most ten rungs, and always appends 30k. Accuracy and
parameter count remain the official qualification and objective; rung choice
cannot redefine success.

Training-rung receipts record every stage, metric, time, failure, accepted or
fallback candidate policy, highest level, and qualification level.

### Tiny AdderBoard

The four-digit task uses a single uninterrupted evaluator-owned trajectory with
default checks at 200, 400, 600, and 1,000 optimizer steps. It stops at the
first rung reaching 99% exact public accuracy. The candidate may edit a literal
`EVALUATION_LADDER`, but the evaluator enforces 100-1,000 steps, at most six
rungs, and the common 1,000-step terminal check. Unlike the ten-digit bridge,
no prefix is retrained for a later rung.

Training/public/holdout examples occupy disjoint deterministic hash buckets.
Every evaluation begins from fresh initialization and uses protected generic
autoregressive decoding, learned-attention execution and ablation checks, and a
25,000-parameter ceiling. A pre-launch MPS timing gate compares its seed
evaluation with the Fashion-MNIST seed evaluation under the same idle host
conditions; the Tiny task must fall within the declared matching tolerance
before an official campaign is launched.

## Developmental value without hidden selection changes

Every proposal receives a separate developmental assessment:

- `primary_retained`;
- `provisional_valid`;
- `provisional_screened`;
- `provisional_nonqualifying`;
- `rejected`.

Credit components record clean execution, a new semantic delta/mechanism,
proximity to the incumbent or qualification boundary, and primary retention.
Thus an executable 98% AdderBoard candidate can remain useful evidence without
becoming a parent. Up to eight informative provisional results remain visible
in a bounded evidence archive.

`selection_effect = "none"` is mandatory in this version. Developmental value
changes the evidence available to the researcher but never changes the strict
official parent or incumbent selection rule. Thus the primary search remains
recognizably Autoresearch/OpenEvolve-style optimization while failures can
serve as scientific evidence. A later study may separately randomize whether
provisional candidates can become parents; that would be a search-algorithm
treatment, not a silent implementation detail.

## Outcomes and construct validity

Do not reduce the study to “which prompt got the best score.” Report at least:

1. executable proposal rate and failure categories;
2. primary retention and objective improvement;
3. stated and implemented mechanism evidence;
4. novel semantic-delta rate;
5. intervention-to-valid-implementation attrition;
6. retention within the five-proposal intervention phase;
7. later downstream discovery;
8. developmental/provisional outcomes;
9. training-ladder promotion and full-fidelity confirmation;
10. sealed final performance and robustness where available;
11. input, cached-input, output, and reasoning-token use;
12. evaluator time, wall-clock active time, and physical versus inherited cost;
13. repetition, homogenization, proxy optimization, and reward-gaming cases;
14. negative or harmful intervention effects.

Mechanism reporting is a trace-completeness measure, not proof that the
implementation changed mechanism. Paper claims should use blinded source-delta
review and executable behavior where feasible. The first post-fork phase is
the cleanest causal contrast. Later phases require trajectory-aware or
hierarchical analysis.

The two controls answer different questions. The passive control estimates the
effect of adding any special instruction. The active reflection control helps
separate semantic content from merely prompting the model to spend more effort.
`exploitation_focus` is a substantive local-search policy rather than a neutral
control.

Because 23 arms create many comparisons, this campaign is exploratory. Rank
uncertainty and raw replicate plots are more important than isolated p-values.
Any confirmatory follow-up should preregister a small subset of contrasts on
new seeds and preferably a held-out task family.

## Reliability and independent control

Each arm has an independent desired state in
`semantic-run-control.json`. The campaign orchestrator balances runnable arms
by proposal count. An infrastructure exception pauses only the affected arm;
an exception during a physically shared prefix pauses that replicate's arms
because none can validly fork from an incomplete prefix. Other replicates and
arms continue.

Adding a prompt arm uses a short campaign-metadata write lock. Proposal starts
may wait briefly while the new prompt bundle, run manifests, prefix inheritance,
schedule, and run-control registry are committed. Already-running subject calls
and evaluators do not take that exclusive lock and do not need to drain. The
event-driven scheduler reloads the registry after the transaction and dispatches
the new trajectories independently.

Campaign pause is cooperative: already-started provider/evaluator work finishes
and is charged before the orchestrator exits. Resume uses the existing artifacts
and recovery path. No completed opportunity is deleted or retried.

## Prepared task/framework matrix

The same engine accepts any v3 `TaskSpec` and `FrameworkSpec`:

| Task | Greedy OpenEvolve | Native OpenEvolve | Autoresearch |
|---|---|---|---|
| Fashion-MNIST | `fashion_mnist_openevolve_v3.toml` | `fashion_mnist_native_openevolve_v3.toml` | `fashion_mnist_autoresearch_v3.toml` |
| AdderBoard | `openevolve_v3.toml` | `native_openevolve_v3.toml` | `autoresearch_v3.toml` |
| nanoGPT | `nanogpt_openevolve_v3.toml` | `nanogpt_native_openevolve_v3.toml` | `nanogpt_autoresearch_v3.toml` |

The first launched campaign used Fashion-MNIST/Greedy OpenEvolve. Preparing a
configuration does not start a subject call.

## Commands

From the repository root:

```bash
PY=architecture_discovery/.venv/bin/python
EXP=experiments/semantic_intervention_experiment.py
OVERNIGHT=experiments/semantic_intervention_overnight.py
PROTO=experiments/c0c3_factorial/configs/protocols/semantic_interventions_v4.toml
TASK=experiments/c0c3_factorial/configs/tasks/fashion_mnist_semantic_v4_mps.toml
FRAMEWORK=experiments/c0c3_factorial/configs/frameworks/fashion_mnist_openevolve_v3.toml
ARMS=experiments/c0c3_factorial/configs/interventions/semantic_research_v4.toml
CAL=data/c0c3/semantic-interventions-v4-fashion-calibration
CAMPAIGN=data/c0c3/semantic-interventions-v4-fashion-openevolve-campaign

$PY "$EXP" prepare \
  --protocol "$PROTO" \
  --task "$TASK" \
  --framework "$FRAMEWORK" \
  --interventions "$ARMS" \
  --calibration "$CAL" \
  --output "$CAMPAIGN" \
  --python-bin "$PY"

$PY "$EXP" validate --campaign "$CAMPAIGN"

$PY "$OVERNIGHT" start \
  --campaign "$CAMPAIGN" \
  --runtime-root "$PWD" \
  --python-bin "$PY" \
  --max-workers 0
```

The command above reproduces the Greedy OpenEvolve architecture used by the
existing 69-trajectory campaign. For a new campaign backed by OpenEvolve's
native population engine, change only `FRAMEWORK` and use a distinct output:

```bash
FRAMEWORK=experiments/c0c3_factorial/configs/frameworks/fashion_mnist_native_openevolve_v3.toml
CAMPAIGN=data/c0c3/semantic-interventions-v4-fashion-native-openevolve-campaign
```

Preparation and validation do not start any trajectory.

The Fashion-MNIST preset runs each trajectory for 200 proposals. To extend an
already drained semantic campaign to the same total without discarding prior
work, use:

```bash
$PY "$EXP" extend-budget --campaign "$CAMPAIGN" --proposals 200 \
  --reason "operator-authorized 200-proposal trajectory budget"
```

Status and cooperative campaign controls:

```bash
$PY "$OVERNIGHT" status --campaign "$CAMPAIGN"
$PY "$OVERNIGHT" pause --campaign "$CAMPAIGN" --reason "operator pause"
$PY "$OVERNIGHT" resume --campaign "$CAMPAIGN" --runtime-root "$PWD" --python-bin "$PY" --max-workers 0
```

Control one arm, then resume the campaign process if it has exited:

```bash
$PY "$EXP" control-run --campaign "$CAMPAIGN" \
  --run-id <exact-run-id> --desired paused --reason "operator pause"

$PY "$EXP" control-run --campaign "$CAMPAIGN" \
  --run-id <exact-run-id> --desired running --reason "operator resume"
```

Launch the live dashboard with the semantic campaign path:

```bash
$PY experiments/live_trajectory_dashboard.py \
  --semantic-v4-fashion-mnist-campaign "$CAMPAIGN"
```

The server is read-only. Open `http://127.0.0.1:8765/` and use **Refresh now**.

## Launch gates

Before an official launch:

1. all prior AdderBoard supervisors must show every run desired paused and no
   active opportunities;
2. the exact runtime must be committed and copied to a detached runtime root;
3. Ruff and the full test suite must pass in both repository Python environments;
4. a disposable campaign must prove literal prefix sharing, five-proposal
   session reset, prompt divergence, individual pause/resume, crash recovery,
   resource de-duplication, and failure isolation;
5. Fashion-MNIST screening thresholds must be checked empirically against the
   seed and representative valid candidates;
6. the official campaign must validate from the detached runtime;
7. the dashboard API and page must be tested against that campaign;
8. AdderBoard and Autoresearch campaigns must remain unstarted.
