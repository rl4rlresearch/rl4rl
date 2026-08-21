# Research-process intervention experiments

This package studies how visible memory and assumption-challenge prompts change
the reasoning dynamics of existing research controllers. It does not introduce
a new controller and does not optimize final benchmark performance.

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
