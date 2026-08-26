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

- 21 intervention arms.
- 3 independently seeded replicates per arm.
- 63 logical trajectories.
- 40 proposal opportunities per trajectory.
- One literal five-proposal prefix shared by all arms within a replicate.
- The first intervention occurs at proposal 6; later interventions occur at
  11, 16, 21, 26, 31, and 36.
- A Codex conversation persists for five proposals, then a new conversation
  begins from the bounded, verified evidence capsule and filesystem state.
- The same replicate seed, starting source, evaluator, model, reasoning effort,
  public evidence policy, proposal budget, and maximum evaluator budget are
  used across arms.

The shared prefix requires only 15 physical proposal calls. Post-fork branches
require 2,205 calls, for 2,220 physical calls and 2,520 logical trajectory
records. Shadow-prefix records are explicitly marked so campaign cost is not
counted 21 times.

Three replicates are intentionally modest. This campaign prioritizes breadth
for exploratory treatment screening. Proposals are repeated observations
within a trajectory, not independent replicates. Results must show all three
replicates and effect sizes; they cannot turn 40 proposals into `n=40`.

## Intervention arms

| ID | Family | Research operation |
|---|---|---|
| `passive_control` | control | No extra phase-boundary direction |
| `active_reflection_control` | control | Effort-matched generic evidence review |
| `assumption_challenge` | epistemic | Identify and test a load-bearing assumption |
| `alternative_mechanism` | epistemic | Implement a materially different learned computation |
| `failure_diagnosis` | epistemic | Diagnose causal failure before changing the design |
| `evidence_contradiction` | epistemic | Center the result hardest to reconcile with the current account |
| `counterfactual_design` | epistemic | Design as if the incumbent were unavailable |
| `falsification` | epistemic | Seek a discriminating result that could refute the current claim |
| `experimental_redesign` | methodological | Redesign the next experiment to isolate an uncertainty |
| `causal_attribution` | methodological | Isolate the cause of a recent improvement or regression |
| `bottleneck_diagnosis` | strategic | Identify and attack the current limiting mechanism |
| `competing_hypotheses` | epistemic | Choose a test that separates competing explanations |
| `information_gain` | methodological | Optimize the next trial for decision-relevant information |
| `ablation_first` | methodological | Remove or control a component to test what is load-bearing |
| `representation_reframe` | epistemic | Change the learned representation of the computation |
| `mechanism_recombination` | strategic | Cleanly combine complementary supported mechanisms |
| `constraint_inversion` | strategic | Reverse a habitual priority to expose a neglected mechanism |
| `exploitation_focus` | strategic control | Make the strongest evidence-led local refinement |
| `uncertainty_calibration` | epistemic | Test the consequential claim with greatest uncertainty |
| `assumption_plus_falsification` | limited combination | Challenge an assumption through a falsifiable alternative |
| `diagnosis_plus_redesign` | limited combination | Diagnose failure and construct a discriminating redesign |

The prompt text is frozen into each created campaign with file and tree hashes.
The labels above support analysis; the subject sees only the task, its available
design evidence, the normal research contract, and the relevant direction.

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

## Multi-fidelity evaluation

### Fashion-MNIST

The default ladder presents 25,000, 50,000, then 100,000 training examples.
Weak candidates may stop after a screening rung. A candidate cannot replace
the incumbent unless it reaches and passes the common 100,000-example
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
at the first rung reaching 99% accuracy and otherwise rejects after 30k. An
agent may add or edit a literal `EVALUATION_LADDER` in editable
`src/train.py`. The evaluator constrains it to 1k-30k, at most ten rungs, and
always appends 30k. Accuracy and parameter count remain the official
qualification and objective; rung choice cannot redefine success.

Training-rung receipts record every stage, metric, time, failure, accepted or
fallback candidate policy, highest level, and qualification level.

## Developmental value without hidden selection changes

Every proposal receives a separate developmental assessment:

- `primary_retained`;
- `provisional_valid`;
- `provisional_screened`;
- `rejected`.

Credit components record valid execution, a new semantic delta/mechanism,
proximity to the incumbent, and primary retention. Up to eight informative
provisional results remain visible in a bounded evidence archive.

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

Because 21 arms create many comparisons, this campaign is exploratory. Rank
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

Campaign pause is cooperative: already-started provider/evaluator work finishes
and is charged before the orchestrator exits. Resume uses the existing artifacts
and recovery path. No completed opportunity is deleted or retried.

## Prepared task/framework matrix

The same engine accepts any v3 `TaskSpec` and `FrameworkSpec`:

| Task | OpenEvolve framework | Autoresearch framework |
|---|---|---|
| Fashion-MNIST | `fashion_mnist_openevolve_v3.toml` | `fashion_mnist_autoresearch_v3.toml` |
| AdderBoard | `openevolve_v3.toml` | `autoresearch_v3.toml` |
| nanoGPT | `nanogpt_openevolve_v3.toml` | `nanogpt_autoresearch_v3.toml` |

Only Fashion-MNIST/OpenEvolve is authorized for the first launch. Preparing a
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
  --max-workers 12
```

Status and cooperative campaign controls:

```bash
$PY "$OVERNIGHT" status --campaign "$CAMPAIGN"
$PY "$OVERNIGHT" pause --campaign "$CAMPAIGN" --reason "operator pause"
$PY "$OVERNIGHT" resume --campaign "$CAMPAIGN" --runtime-root "$PWD" --python-bin "$PY" --max-workers 12
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

