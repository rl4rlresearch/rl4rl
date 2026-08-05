# AdderBoard Autonomous-Research Trajectory Analysis

This package measures the research behavior of autonomous discovery systems
while they attempt the same operational task: minimize parameter count subject
to at least 99% exact-match accuracy on AdderBoard. The paper-level question is
not just “which system won?” It is “what kinds of search moves did each system
make, which moves survived evaluation, when did its search become repetitive,
and what did it regard as a plausible stopping point?”

OpenEvolve, Autoresearch, and TTT-Discover produce different native logs. The
`trajectory_analysis` package converts them to one loss-minimizing event model,
keeps a cryptographic reference to every raw record, validates causal ordering,
joins double-coded architecture annotations, computes predeclared run-level
metrics, and emits a reproducible report bundle.

## Scientific boundary

One complete, independently launched search run is the independent unit. A
candidate, mutation, rollout, or commit is a within-run observation and must not
be counted as an independent replicate. Cross-paradigm outputs are descriptive;
the pipeline never silently manufactures p-values or confidence intervals from
candidate-level pseudo-replication.

The active protocol is
[`../TRAJECTORY_STUDY_PROTOCOL.md`](../TRAJECTORY_STUDY_PROTOCOL.md). Claims are
limited to the observed AdderBoard setup. A single synthetic benchmark cannot
establish a universal law of autonomous scientific reasoning.

## Install and validate

```bash
git submodule update --init --recursive
uv sync --python 3.12
.venv/bin/python -m compileall -q trajectory_analysis scripts tests
.venv/bin/python -m pytest -q
```

Run the end-to-end provider-free smoke in a path that does not yet exist:

```bash
.venv/bin/python scripts/trajectory_offline_smoke.py \
  --output-dir /private/tmp/rl4rl-trajectory-smoke
```

It generates labeled synthetic examples for all three adapters, validates
lineage and hashes, exercises adjudication, calculates metrics, and renders the
full report. The output and its `-inputs` sibling are synthetic fixtures only.

## Prepare real inputs

Copy `trajectory_study_manifest.template.yaml` outside the repository or into a
private data directory. Do not edit the source exports after hashing them.

Each source is one independent run and selects one adapter:

| Adapter | Format | Required identity/order fields | Common metric fields |
| --- | --- | --- | --- |
| `autoresearch_tsv_v1` | headered TSV | `sequence`, `commit`, `parent_commit` | `accuracy`, `parameters`, `valid`, `status` |
| `openevolve_jsonl_v1` | one JSON object per line | `iteration`, `candidate_id`, `parent_ids` | `accuracy`, `parameters`, `valid`, `accepted` |
| `ttt_jsonl_v1` | one JSON object per line | `step`, `rollout_id`, `parents` | `exact_match`, `num_params`, `is_valid`, `selected` |

Aliases accepted by the adapters are documented in
`trajectory_analysis/adapters.py`. Unknown native fields are retained under
event `metadata`; raw rows are never rewritten. Accuracy must already be a
fraction in `[0, 1]`—the adapter will not guess whether `99` means 99 percent.

Calculate hashes on macOS with:

```bash
shasum -a 256 path/to/source.jsonl
shasum -a 256 path/to/annotations.jsonl
```

The manifest rejects absolute paths, parent-directory escapes, unsupported
adapters, duplicate run/source IDs, absent files, and hash mismatches.

## Annotate architecture moves

Only events with at least one parent are treated as edits. Each edit needs
exactly two independent `coder` records. If either the edit family or boundary
class differs, exactly one `adjudicator` record is required. See
`trajectory_annotation_codebook.yaml` for allowed labels and decision rules.

Annotation files are JSONL:

```json
{"event_id":"oe-01:7","annotator_id":"coder-a","role":"coder","edit_family":"positional","boundary_class":"ontology_changing","rationale":"Replaces learned positions with a rotary computation."}
```

Keyword-based suggestions are written separately to
`annotation_suggestions.jsonl`. They are annotation aids only and are never
promoted to scientific labels.

## Validate before analysis

```bash
.venv/bin/python scripts/trajectory_validate.py \
  --manifest /path/to/manifest.yaml \
  --data-root /path/to/frozen-data
```

Validation is read-only. It verifies hashes, schemas, finite metrics, unique
event sequences, single-paradigm runs, parent-before-child ordering, terminal
stop events, annotation coverage, independence of coders, and adjudication.

`--allow-unannotated` exists only for engineering inspection. It marks missing
edit labels `unclassified`; do not use that option for paper results.

## Run the analysis

The output directory must not exist, preventing accidental overwrites:

```bash
.venv/bin/python scripts/trajectory_analyze.py \
  --manifest /path/to/manifest.yaml \
  --data-root /path/to/frozen-data \
  --output-dir /path/to/new-analysis-directory
```

The bundle contains:

- `normalized_events.jsonl` with raw-record hashes and resolved labels;
- `run_summaries.{json,csv}` and `paradigm_summaries.{json,csv}`;
- `run_contexts.{json,csv}` and `comparability_warnings.json` for generator,
  evaluator, prompt, tools, start state, seeds, budgets, and archive reuse;
- `annotation_agreement.json` with exact agreement and Cohen’s κ;
- `lineage_edges.csv` and `lineage.dot`;
- `frontier_progression.svg`, `boundary_edit_mix.svg`, and
  `rolling_architecture_diversity.svg`;
- `report.md` with claim limits and independent-unit warnings;
- `provenance.json` with every frozen input and generated-output digest.

Run-level metrics include qualifying frontier improvements, gap to the frozen
external frontier, edit-family entropy, ontology-changing attempt/accept/qualify
rates, acceptance/invalid/rollback/revisit rates, fingerprint coverage, longest
ontology-preserving streak, time to first boundary-crossing move, invalid-parent
to valid-child events, and premature optimality claims.

## What remains data-blocked

The repository does not contain the original OpenEvolve, Autoresearch, or
TTT-Discover trajectories described in the project note. It therefore makes no
real comparative findings. Before a paper run, obtain complete exports (not
curated winners), record prompts/tools/budgets/environment and all exclusions,
freeze hashes, double-code edits blind to system identity when feasible, and
record which runs are genuinely independent.

## Retained prospective apparatus

The existing `study`, `evaluation`, `sealed_eval`, `containment`,
`architecture_ir`, `novelty`, `review`, `mechanism`, `replication`, `analysis`,
`reconstruction`, `research_ledger`, and `reporting` packages implement the
earlier C0–C3 prospective novelty experiment. They remain tested and useful for
future controlled follow-ups, but they do not define this branch’s primary
research question and should not be mixed into the retrospective trajectory
analysis without a protocol amendment.
