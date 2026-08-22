# Research-process intervention experiments

This package studies how visible memory and assumption-challenge prompts change
the reasoning dynamics of existing research controllers. It does not introduce
a new controller and does not optimize final benchmark performance.

## Executive summary

The central question is not simply whether an intervention produces a better
final model. It is whether changing what a research agent remembers and asking
it to challenge a consequential assumption changes the *path of research*: the
hypotheses it maintains, the experiments it chooses, how far it moves after new
evidence, which failed ideas it revisits, and whether it uses contradictory
results to revise its explanation.

The implementation wraps the repository's existing AutoResearch and OpenEvolve
controllers with an opt-in, process-observation layer. It holds evaluation,
training, rewards, selection, compute, and stopping rules fixed. The layer only:

1. controls which public prior results are visible in a bounded memory packet;
2. switches between neutral review and a frozen assumption-challenge prompt;
3. asks for short, inspectable lab-note metadata; and
4. records treatment exposure, decisions, lineage, and public outcomes.

On 2026-08-21/22, the complete eight-run engineering pilot finished on Modal:
two frameworks crossed with four treatment cells, four proposal opportunities
per run, and 32 total proposal opportunities. All eight final runs returned
code 0, produced distinct artifact manifests, and have successful immutable
terminal receipts. This establishes that the experimental machinery is usable;
it is not yet enough replication for a causal or scientific result.

## Research question and contribution

### Main question

How do bounded portfolio memory and scheduled assumption challenges, separately
and together, change the research dynamics of LLM-based research controllers?

The intended paper contribution is a process-level causal design and
instrumentation system. It treats the sequence of ideas, explanations,
experiments, evidence updates, and selection decisions as the object of study.
Final benchmark performance is a downstream descriptive outcome, not the
primary target.

### Why this is important

Research agents can arrive at a strong or weak final score for many accidental
reasons. A final score alone does not tell us whether an agent explored several
mechanisms, learned from disconfirming evidence, repeatedly polished one idea,
or changed explanations without acknowledging the change. If autonomous
research systems are used to generate scientific claims, their evidence-use and
search dynamics need to be observable and experimentally testable.

### Why this is interesting

The interventions may have non-monotonic and framework-dependent effects.
Portfolio memory could preserve abandoned alternatives, or it could anchor the
agent to a fixed menu. Assumption challenges could produce discriminating
experiments, or merely generate plausible-sounding criticism. Their combination
could create productive recombination, overwhelm the prompt, or behave
differently in a greedy controller and a population-based controller. These
interactions are scientifically richer than asking whether a longer prompt
increases a benchmark score.

## Experimental design

This is a 2 × 2 factorial design within each framework.

| Condition | Visible memory | Deliberation | Intuitive description |
|---|---|---|---|
| `RD0` | Sequential | Neutral review | Show the most recent direction and ask what to try next. |
| `RD1` | Sequential | Assumption challenge | Show the most recent direction and require a consequential assumption test. |
| `RD2` | Four-slot portfolio | Neutral review | Show several strategically different prior directions and ask what to try next. |
| `RD3` | Four-slot portfolio | Assumption challenge | Show the portfolio and require an experiment that distinguishes explanations. |

The two factors are:

- **Memory policy:** whether the controller sees only the current/recent entry
  or a fixed four-slot portfolio of public prior work.
- **Deliberation policy:** whether it receives a neutral evidence-review prompt
  or an assumption-challenge prompt at precommitted opportunities.

The framework axis contains:

- **AutoResearch:** the existing greedy, sequential controller.
- **OpenEvolve:** the existing population/portfolio controller.

Framework is a moderator and replication axis in this pilot, not a randomized
treatment. The visible-memory intervention does not change either framework's
native parent selection, population management, reward, or retention logic.

### Intuitive example

Suppose the current explanation is “adding a gated residual path improves
carry propagation.” A sequential condition shows only that recent direction.
A portfolio condition may additionally show a valid failure, a lexically
distant alternative such as a recurrence mechanism, and an abandoned direction.

The neutral prompt asks the agent to review the evidence and choose its next
experiment. The challenge prompt instead asks which unsupported claim would
most change the decision, requires an alternative explanation, and asks for a
test whose public result would distinguish the two. For example, the agent may
contrast “gating improves carry propagation” with “the improvement only comes
from added depth,” then propose an ablation that matches depth while removing
the gate.

The experiment studies whether this intervention actually changes what the
agent does and how it reacts to the result—not whether the resulting prose
sounds more critical.

## Exact intervention mechanics

### Bounded visible memory

Every memory packet is padded or truncated to exactly 6,000 characters. It may
use only public evaluation fields: execution success, transformer validity,
public accuracy, search score, parent eligibility, and failure stage. It cannot
read sealed outcomes or future results.

Sequential memory fills only `current_or_recent`. Portfolio memory uses four
fixed semantic slots:

1. `current_or_recent`: the latest retained direction, or latest entry;
2. `valid_failure`: a rejected candidate that executed successfully when one
   exists;
3. `distant_alternative`: the prior entry with greatest lexical distance from
   the current mechanism description; and
4. `abandoned_direction`: another rejected direction when available.

Unavailable slots are explicit placeholders. This keeps the schema fixed and
prevents the number of displayed entries from silently revealing information.

### Assumption challenge

The challenge instruction asks the controller to identify one claim that the
current direction treats as true without decisive public evidence. It must pick
the claim whose rejection would most change the next decision, state an
alternative explanation, identify evidence favoring each explanation, and
choose an experiment because it distinguishes them. Cosmetic challenges do not
satisfy the instruction.

Neutral and challenge blocks are both padded to 1,800 characters. Provider token
counts are recorded separately because equal characters do not guarantee equal
tokens. In the completed four-opportunity pilot, challenge conditions were
precommitted to opportunities 1, 2, 3, and 4; neutral conditions had no
challenge opportunities.

### Public lab notes

Each proposal is asked to include one or two auditable sentences for the current
explanation, supporting evidence, next experiment, expected result, decision
rule, interpretation of the previous result, whether the result changed the
explanation, challenged assumption, alternative explanation, and
discriminating evidence. These are top-level candidate metadata and do not alter
the executable architecture. They are not private chain-of-thought.

### One opportunity, step by step

1. The wrapper reads only eligible prior public records.
2. It deterministically constructs the sequential or portfolio memory packet.
3. It appends the neutral or scheduled challenge block.
4. The unchanged controller proposes a complete Architecture IR and public lab
   note.
5. The unchanged evaluator trains/evaluates the candidate and emits public
   feedback.
6. The unchanged native selection logic accepts, rejects, archives, or replaces
   the candidate.
7. The wrapper records exposure, proposal, parent, result, retention decision,
   and the following interpretation in hash-bound artifacts.

## What is held constant

Within the pilot, all final cells use the same:

- starting checkpoint, SHA-256
  `b9020bc633d9d999751dcbf1101e636bf034be278367fa7774cfd992b6764bcb`;
- model, `gpt-5.6-sol`, with high reasoning effort;
- four proposal opportunities and at most one provider request per opportunity;
- seed 1, zero provider retries, and zero Modal retries;
- `smoke_train_cuda_v2` training and smoke evaluation on one NVIDIA T4;
- complete Architecture IR proposal format;
- 16,384 maximum completion tokens per request;
- public evaluator feedback, reward, eligibility, native parent selection,
  archive/retention rules, stopping rule, and compute profile.

The randomized launch order is frozen in the launch manifest. This small pilot
has one run per framework × condition cell, so order randomization prevents an
obvious systematic ordering choice but cannot by itself create adequate
replication.

## Completed Modal engineering pilot

### Frozen identities

- Final local commit: `7498a662c4283eaf756ef814875842ed7d7bdd59`
- Source-tree SHA-256:
  `ea6d925d495a2c2ba3f8dc49c2ab873346607d8c939944e467f4bbe597aee087`
- Modal image-source SHA-256:
  `56d1976d9fd27eac5c108af86d2269d91fdb632234e8117e014f8e26f8271e7e`
- Modal image ID: `im-myPyiMq25YHvFcT8qBP1Db`
- Cohort: `modal-cuda-env-20260821-rp5`
- Modal profile/environment: `scalingintelligence` / `main`
- Artifact volume: `rl4rl-architecture-artifacts`
- Claim scope: `non_scientific_engineering_pilot`

Before experiments, 1,563 repository checks passed and were sealed in local
engineering-freeze identity
`1d4b9279cf8ec11303b0773ef8ab8a54c1a3dcff7ec3f968bfd911f9c4275fb1`.
Eight provider-free Modal readiness actions then verified CUDA, offline smoke
execution, candidate training, artifact round trips, and checkpoint resume.
The final candidate/resume preflight binding was
`aae9061eace3d591057ac423199d873e9306942e23ad737b5f044f7cc73b2db8`.

### Final execution order and outcomes

| Position | Framework | Cell | Final run ID | Attempt ID | Artifact manifest SHA-256 | Status |
|---:|---|---|---|---|---|---|
| 1 | OpenEvolve | `RD1` | `oe-modal-rd1-20260821-06` | `8687ed8134bfdcc2afc9c4310d419b7c` | `7553c4e5c47db998146a7896cfc2daef6a7f0034a2deca9ef67b77cbbbc8722f` | succeeded |
| 2 | AutoResearch | `RD0` | `ar-modal-rd0-20260821-03` | `5d655f6b93b21ade9d9f0abdb63fc678` | `4ceb18479977af9eb83d1cca04368e9e350441c3b0347e91a0fccbb57193fbc0` | succeeded |
| 3 | OpenEvolve | `RD0` | `oe-modal-rd0-20260821-02` | `85ca56a2afc4c03e5027e8a2b62135bc` | `40665d00a3ce3583776106e1e372b8ea496b4d3e7c07113111035b1c23fe5d76` | succeeded |
| 4 | OpenEvolve | `RD3` | `oe-modal-rd3-20260821-02` | `9e059614ba106d12cecd64eca86de0d6` | `72b5a16dcf86d80f473317e6a6dbaa9580b47bdade59289bcfe182c55a8194b5` | succeeded |
| 5 | OpenEvolve | `RD2` | `oe-modal-rd2-20260821-02` | `86d6749c3a66f9617b0b5226d340f176` | `84dbcb5c00e8821af7f8dcc4da714d30ca38f9824817f46ee6ad65d01bb16eeb` | succeeded |
| 6 | AutoResearch | `RD2` | `ar-modal-rd2-20260821-02` | `ee9d4f67ad70171e3d08e4a390aa8957` | `27f33c2fdedd2fed3f8ad6e36b312dad7fa879eb0f727a9f32812f132869624b` | succeeded |
| 7 | AutoResearch | `RD1` | `ar-modal-rd1-20260821-02` | `f64535a3badc4962397fb67e7f66a5f2` | `e49b947e42002660d22e2b3caf27387c733d99759b60f899859fae4ff1c66673` | succeeded |
| 8 | AutoResearch | `RD3` | `ar-modal-rd3-20260821-02` | `649ed80a34b3fee338e464a36d8bad44` | `81a2ab0da68526805828ca20b0a06dfb2629b65219189b95fc7fcaef63479eb2` | succeeded |

All eight have exactly one final terminal receipt in the final cohort, status
`succeeded`, and return code 0. The receipt schema conservatively records that
remote execution “may have started”; this is normal containment language and
does not contradict the explicit successful status and return code.

Each remote artifact root is:

```text
volume://rl4rl-architecture-artifacts/runs/<final-run-id>
```

Local action receipts are generated under:

```text
outputs/readiness/modal_only_final/modal_live_cohorts/
  ea6d925d495a2c2ba3f8dc49c2ab873346607d8c939944e467f4bbe597aee087/
  56d1976d9fd27eac5c108af86d2269d91fdb632234e8117e014f8e26f8271e7e/
  modal-cuda-env-20260821-rp5/action_attempts/
```

`outputs/` is intentionally ignored by Git. The hashes and run IDs above are
the durable lookup keys; teammates in the Modal workspace can download the
corresponding volume artifacts through the repository's bound download path.

### Authorization envelopes

These are local approval ceilings, not measured charges or platform-enforced
billing limits:

- eight provider-free readiness actions: `$0.64551825` total Modal approval;
- each experiment: `$0.33603658125` Modal and `$22.9376` provider approval;
- eight experiments: `$2.68829265` Modal and `$183.5008` provider approval;
- maximum provider requests: 32.

Actual billing may be lower or differ. No cost conclusion should be inferred
from these local gates.

## What the completed pilot establishes

The pilot establishes engineering facts:

- all four treatment configurations can be attached to both controllers;
- treatment payloads survive the local-to-Modal boundary;
- the same source, image, checkpoint, seed, and budget can execute all cells;
- public exposure and decision ledgers are emitted, including the intentionally
  empty control decision ledger where no deliberation record exists;
- provider-attempt accounting, terminal receipts, and artifact manifests seal;
- the Mac launcher's containment evidence survives a legitimate wall-clock
  correction without confusing it with an OS reboot.

The pilot does **not** establish that either intervention improves diversity,
scientific reasoning, or final performance. There is only one short trajectory
per cell, the training/evaluation profile is explicitly non-scientific, and the
process artifacts have not yet undergone blinded semantic annotation.

## Analysis plan

### Primary process outcomes

The primary paper-facing outcomes are:

1. **Discriminating-experiment rate:** how often the chosen experiment can
   separate the stated explanation from a concrete alternative.
2. **Research displacement:** how far the next move departs from the current
   hypothesis/mechanism, from `D0` (same hypothesis) through `D5` (problem
   reformulation).
3. **Evidence-responsive revision after contradiction:** whether a contradicted
   prediction leads to weakening, rejection, narrowing, replication, or an
   auxiliary explanation rather than silent continuation.

### Secondary outcomes

- research-move and epistemic-purpose entropy;
- transition matrices and move persistence;
- hypothesis lifetime;
- rationale/action alignment;
- whether the interpretation is supported by the public result;
- challenge uptake and visible-memory citation;
- lineage branching and parent concentration;
- lexical explanation/experiment diversity as an immediate diagnostic only;
- final public score as a separate downstream descriptive outcome.

### Annotation and inference

Semantic claims should use blinded annotation. Annotators receive local decision
context but not framework, treatment, final score, or future success. Report
inter-annotator agreement and adjudicate disagreements under the frozen
codebook. Do not treat the four decisions within a trajectory as independent
replicates. Checkpoint-fork experiments should use paired contrasts or
checkpoint fixed effects; full trajectories should use run-level or
block-aware uncertainty.

The factorial contrasts of interest are the memory main effect, challenge main
effect, memory × challenge interaction, and moderation by framework. With one
pilot run per cell these contrasts are descriptive only. A scientific follow-up
needs multiple randomized blocks/checkpoints and a precommitted analysis plan.

## Failure history and fixes

Several failed or superseded launch attempts were useful engineering evidence
but are not members of the final cohort:

- early attempts exposed missing frozen source-identity manifests;
- an expired/exhausted OpenAI credential produced genuine provider HTTP 429
  failures until the Modal secret was corrected;
- full evolution validation initially expected a canary-only generator source;
- the provider ledger validator initially compared CLI action `evolve` with the
  remote action name `evolution_run`;
- the RD0 control could legitimately emit an empty `decisions.jsonl`, which a
  generic non-empty-tree validator originally rejected;
- macOS kept the same boot UUID while correcting `kern.boottime` by one second,
  causing a completed remote run to fail local terminal-receipt persistence.

The final launcher treats the kernel boot UUID as authoritative for same-session
identity while retaining the validated boot timestamp to order genuinely
different boot UUIDs. The focused launcher tests and the complete 1,563-test
suite passed after this change. The final authoritative cohort is `rp5`; do not
combine earlier run IDs with the final eight-run table.

## Implementation map

- `research_dynamics/contracts.py`: treatment, exposure, decision, and lab-note
  schemas.
- `research_dynamics/memory.py`: public-field allowlist and deterministic
  sequential/portfolio memory selection.
- `research_dynamics/prompts.py`: frozen, character-matched neutral and
  assumption-challenge instructions.
- `research_dynamics/extraction.py`: extraction of public lab notes and linked
  decision records.
- `research_dynamics/orchestration.py`: fork/full planning and fresh-output
  execution.
- `research_dynamics/openevolve_integration.py`: scoped instrumentation inside
  OpenEvolve proposal workers.
- `research_dynamics/annotations.py`, `codebook.py`, and `metrics.py`: blinded
  annotation export and process-first summaries.
- `agents/greedy_autoresearch/run.py` and
  `agents/semantic_autoresearch/run.py`: opt-in AutoResearch integration.
- `common/openevolve_runner.py`: opt-in OpenEvolve integration.
- `common/evolution_run.py`, `modal_app.py`, `evolve`, and
  `scripts/launch_modal.py`: frozen payload transport, Modal execution, approval
  gates, and receipts.

Ordinary runs without `RL4RL_PROCESS_CONFIG` follow their original controller
path.

The randomized variables are:

- visible memory: sequential (`RD0`, `RD1`) or four-slot portfolio (`RD2`, `RD3`);
- deliberation: neutral evidence review (`RD0`, `RD2`) or a challenge at frozen
  opportunities (`RD1`, `RD3`).

The evaluator, rewards, public feedback, parent sampling, eligibility rules,
archive replacement, compute budget, and stopping rule remain unchanged. The
existing `study.ConditionId` C0-C3 contract also remains unchanged; `RD0`-`RD3`
live separately to prevent semantic collisions.

## Recorded process data

Every treated prompt asks for short public lab-note fields in candidate metadata:
the current explanation, supporting evidence, next experiment, expected result,
decision rule, interpretation of the previous result, whether that result changed
the explanation, challenged assumption, alternative explanation, and evidence
that would distinguish the explanations.

These fields are descriptive metadata. They do not affect executable
architecture identity or selection. They are not requests for private
chain-of-thought. Missing fields remain missing during retrospective import.

Each run writes:

```text
controller_run/
  research_process/
    study_config.json
    exposures.jsonl
    decisions.jsonl
```

`exposures.jsonl` records the active treatment and public memory entries.
`decisions.jsonl` links proposals, public results, retention decisions, and the
following step's interpretation of each result.

## Setup

From `architecture_discovery/`:

```bash
uv sync --offline
```

Omit `--offline` only if the pinned packages are not cached. Provider credentials
and accelerator setup follow the existing controller documentation. This package
adds no external dependency.

## E2: matched checkpoint forks

Use one candidate Architecture IR file as the checkpoint. The planner hashes it,
creates all four branches, randomizes execution order, and freezes a challenge at
every branch decision. The command is an argv JSON array, not a shell string.

AutoResearch:

```bash
python scripts/research_process.py plan-forks \
  --study-id ar-fork-pilot \
  --framework autoresearch \
  --checkpoint /absolute/path/checkpoint.ir.json \
  --output-dir outputs/process/ar-fork-pilot \
  --horizon 4 \
  --seed 1201 \
  --command-json '["python","agents/greedy_autoresearch/run.py","--iterations","{horizon}","--seed","{seed}","--output-dir","{output_dir}","--initial-candidate","{checkpoint}","--engineering-pilot"]'
```

OpenEvolve:

```bash
python scripts/research_process.py plan-forks \
  --study-id oe-fork-pilot \
  --framework openevolve \
  --checkpoint /absolute/path/checkpoint.ir.json \
  --output-dir outputs/process/oe-fork-pilot \
  --horizon 4 \
  --seed 1201 \
  --command-json '["python","agents/openevolve_generic/run.py","--iterations","{horizon}","--seed","{seed}","--output-dir","{output_dir}","--engineering-pilot"]'
```

The executor sets `RL4RL_PROCESS_INITIAL_CANDIDATE` for OpenEvolve, whose native
CLI does not expose an initial-candidate flag. The runner validates this file
before provider initialization.

Inspect commands without running them:

```bash
python scripts/research_process.py run-manifest \
  outputs/process/ar-fork-pilot/fork_manifest.json --dry-run
```

Run all branches:

```bash
python scripts/research_process.py run-manifest \
  outputs/process/ar-fork-pilot/fork_manifest.json
```

The executor re-hashes the checkpoint before execution and refuses non-fresh
branch outputs.

## Modal execution

The existing `evolve` entrypoint now transports an opt-in process assignment to
the existing source-bound `evolution_run` Modal function. It reads the same two
environment variables used by the local executor; no API key is placed in the
payload or inherited by the controller process.

Plan one checkpoint-bound branch without starting paid work:

```bash
RL4RL_PROCESS_CONFIG=/absolute/path/process_config.json \
RL4RL_PROCESS_INITIAL_CANDIDATE=/absolute/path/checkpoint.ir.json \
./evolve openevolve -n 4 --run-id process-oe-rd3-01
```

The plan must report `"process_intervention_attached": true`. After completing
the repository's existing Modal readiness and cost-approval steps, add the
normal execution flags:

```bash
RL4RL_PROCESS_CONFIG=/absolute/path/process_config.json \
RL4RL_PROCESS_INITIAL_CANDIDATE=/absolute/path/checkpoint.ir.json \
./evolve openevolve -n 4 --run-id process-oe-rd3-01 \
  --execute --accept-estimated-cost
```

For a full trajectory without a fork checkpoint, omit
`RL4RL_PROCESS_INITIAL_CANDIDATE`. The remote function validates and
materializes the frozen config (and checkpoint when present) under
`process_inputs/`, passes only those paths to the existing controller, and
publishes `controller/research_process/exposures.jsonl` and
`controller/research_process/decisions.jsonl` with the normal run artifacts.
Each treatment cell remains a separate paid Modal run and therefore retains the
existing per-run approval and cost gates.

## E3: full trajectories

The full planner creates randomized blocks containing one run in each treatment
cell. Freeze the challenge schedule before execution.

```bash
python scripts/research_process.py plan-full \
  --study-id ar-full-v1 \
  --framework autoresearch \
  --output-dir outputs/process/ar-full-v1 \
  --blocks 8 \
  --first-seed 2001 \
  --challenge-schedule 5,10,15,20 \
  --command-json '["python","agents/greedy_autoresearch/run.py","--iterations","24","--seed","{seed}","--output-dir","{output_dir}","--engineering-pilot"]'
```

Use `agents/semantic_autoresearch/run.py`,
`agents/openevolve_generic/run.py`, or
`agents/openevolve_semantic/run.py` to change the experimental subject. Framework
is a moderator/replication axis unless its assignment is randomized separately.

Add `--scientific` only when the underlying command uses the repository's frozen
scientific training/evaluation configuration and authorization records. This flag
labels the process manifest; it does not relax an existing gate.

## E0: retrospective baseline

```bash
python scripts/research_process.py import-baseline \
  --run-dir /absolute/path/existing-run \
  --framework autoresearch \
  --study-id retrospective-v1 \
  --run-id old-run-001
```

The importer uses existing candidate metadata and public results. It does not
invent rationales, assumptions, or interpretations that were not logged.

## Blinded annotation and summaries

Generate immediate annotation-free diagnostics for explanation/experiment
diversity, lineage branching, challenge uptake, and visible-memory citation:

```bash
python scripts/research_process.py summarize-decisions \
  --decisions outputs/process/ar-full-v1/block-000/RD0/controller_run/research_process/decisions.jsonl \
  --output outputs/process/ar-full-v1/block-000/RD0/process_telemetry.json
```

Lexical Jaccard diversity is only a diagnostic. Use blinded semantic annotation
for paper claims about idea or logic diversity.

```bash
python scripts/research_process.py export-annotations \
  --decisions outputs/process/ar-full-v1/block-000/RD0/controller_run/research_process/decisions.jsonl \
  --output-dir outputs/process/ar-full-v1/annotation_batch_01
```

Annotators receive local decision context but not framework, treatment, final
score, or future success. Keep raw annotations separate and adjudicate after
measuring agreement.

```bash
python scripts/research_process.py summarize \
  --annotations outputs/process/ar-full-v1/annotation_batch_01/annotation_template.jsonl \
  --output outputs/process/ar-full-v1/annotation_batch_01/process_summary.json
```

Primary summaries are discriminating-experiment rate, research displacement,
and evidence-responsive revision after contradictory results. Move entropy,
persistence, transition matrices, rationale-action alignment, interpretation
support, and hypothesis lifetime are secondary. Analyze checkpoint forks with
checkpoint fixed effects or paired contrasts; do not treat decisions from one
trajectory as independent replicates. Report final benchmark performance in a
separate downstream section.

## Reproducibility boundaries

- Visible memory allowlists only controller-visible public evaluation fields.
- The planner requires fresh outputs and hashes manifests/configs.
- The executor rejects changed checkpoints and non-fresh run directories.
- OpenEvolve instrumentation is installed inside spawned proposal workers and
  removed afterward.
- A controller without `RL4RL_PROCESS_CONFIG` follows its original prompt path.
- No process command changes selection, evaluation, reward, retention, or stop
  logic.
