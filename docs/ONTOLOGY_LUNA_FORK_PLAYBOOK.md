# Luna campaign fork playbook

## Operator instructions

Use `gpt-5.6-luna` for each campaign fork and all its subagents. This supersedes
the earlier Terra preference for this work. Select Luna for the fork in the app;
prompt text alone does not change the model already running the task.

**Start as many useful subagents as the current runtime permits. Divide work
between yourself and those subagents. Keep available slots occupied while
independent work remains. The main agent must do a share of the work.**

Use small, repetitive assignments with explicit files, IDs, steps and outputs.
Do not delegate "classify the campaign" or "design a better ontology". Use the
existing method and break unfamiliar cases into concrete source-reading tasks.
Keep all source-review and validation requirements. Speed is not permission to
guess labels, call unresolved cases absent, or mark partial work complete.

This playbook governs Addition, nanoGPT, Fashion MNIST, KWS, UCI HAR and
Tiny Adderboard. Tiny Adderboard ends at proposal 100 inclusive. Conditions are C0-C3. Addition ends at proposal
120 inclusive. Each campaign seed is classified once. The family rule remains
exact equality of the complete categorical vector.

## Files and ownership

Let `BASE` be `outputs/ontology-categorical-v1/forks/<campaign>`.
Preparation has already created `queue.json`, `working-schema.json`, a
`packets/seed.json` file and batches of at most ten unique non-seed candidates.
Packets retain every occurrence of each candidate in the trajectory.

| File or directory | Sole writer |
| --- | --- |
| `BASE/progress.json` and `BASE/STATUS.md` | Campaign main agent |
| `BASE/working-schema.json` and `BASE/SCHEMA_CHANGES.md` | Campaign main agent |
| `BASE/seed-parts/<assignment>/` | The worker assigned that seed task |
| `BASE/results/<packet_id>/` | The one agent assigned that packet |
| `BASE/checks/<assignment>/` | The one agent assigned that check |
| `BASE/final.json`, `BASE/UNRESOLVED.md`, `BASE/HANDOFF.md` | Campaign main agent |

Every campaign has its own `BASE`. Workers must not edit other workers' output.
Neither main agents nor workers overwrite the shared schema JSON, shared Python
modules, shared tests, other campaigns, campaign source data, or legacy review
outputs. If a shared-code fix is needed, record a minimal reproducer and a patch
proposal under `BASE`; continue unaffected work. The parent task will integrate
shared changes. Put new campaign-specific checks under `BASE/checks/`.

Campaign schemas remain living. Main agents edit their own `working-schema.json`
as evidence requires and pass it as `schema_override` to `output_document`.
The publisher enforces the original campaign identity and scope. A changed
schema invalidates old review revisions; reproject every prior candidate and
recompute all metrics. A working copy prevents write conflicts, not extensions.

## Startup: do these steps in order

1. Read your campaign kickoff file and the shared method:
   `docs/ONTOLOGY_CATEGORICAL_FINGERPRINT_V1.md` and
   `docs/ONTOLOGY_COMPONENT_REVIEW_RULES.md`.
2. Read `BASE/queue.json` and `BASE/working-schema.json`. Inspect existing results
   and progress. Resume completed work instead of regenerating it.
3. Record the runtime's actual total concurrency allowance. It includes the main
   agent where the runtime says so. For example, four total slots means the main
   agent plus three subagents, not four subagents. Different forks may share
   limits; use the allocation actually granted. Do not change global limits.
4. Create or update `progress.json` with the queue's candidate denominator,
   current schema revision, packet owner, state and result path. Only you write
   this ledger. Assign each packet once before dispatch.
5. Spawn Luna workers to fill currently available slots. With the provided
   collaboration tool, use `model="gpt-5.6-luna"` and `fork_turns="none"`, and
   supply the complete bounded task message. Do not attach the entire long
   conversation. Do not rely on model inheritance from an older Terra session.
6. Start your own independent assignment immediately after dispatch. When a
   worker completes, inspect its output, send its next packet and keep working.
   Reuse workers. Do not repeatedly retry a full allocation. If no slot is
   available, process a packet yourself and reuse capacity when it becomes free.

Only the main agent dispatches workers by default. A worker can request another
bounded task be parallelized; the main agent schedules it within the actual
shared limit. Do not recursively multiply workers or create user-visible tasks.

## First wave: establish the shared seed once

Divide seed preparation into these bounded tasks. Use as many simultaneous
tasks as the runtime allows; finish the remaining tasks yourself or in the next
available slot. Additional free slots can check source availability in separate
candidate packets. These are preparation tasks, not separate seed classifications.

| Assignment | Exact deliverable |
| --- | --- |
| Seed runtime tracing | Read the seed constructor and inference entrypoints. List each reachable helper and an input â†’ operation â†’ output â†’ consumer table with file/line evidence. |
| Seed training tracing | Read the training entrypoint. Identify the objective, optimizer/update law, target transforms and data transforms. Keep numerical settings in evidence. |
| Seed coverage check | Enumerate source statements. Identify inactive code with evidence and any semantic role missing from the working schema. Do not assign guessed categories. |
| Main agent | Load the seed packet, preserve its source bundle, create the component worksheet, and combine the supplied evidence into one seed review. |

Use the same complete source bundle and `source_inventory` IDs in all seed
tasks. Verify files against the candidate artifact, including local imported
helpers and configuration that selects behavior. Windows source paths may need
the existing `_readable_path` helper for long paths.

For each seed component, the main agent applies the component loop below. If a
draft label is provisional, split the problem into the exception tasks below,
then add the precise definition to the working schema. Do not merely remove a
provisional marker. Save one validated canonical seed result under
`BASE/results/seed/`. Give subsequent workers that source-backed example.

## Routine worker: repeat this candidate loop

Work only on the exact candidate IDs in your assigned packet. Complete at most
one ten-candidate packet per assignment, saving after each candidate. If a case
needs new semantic reasoning, return an exception card and continue the other
candidates in the packet. An exception is unfinished work, not a classification.

1. Read the working schema and record `schema_revision(schema)`. Read the
   packet's candidate source locations. Select an available occurrence and
   load its complete source bundle. Do not execute candidate code.
2. Reuse an existing validated review only for the same source digest and schema
   revision. An identical source can reuse the vector, but every trajectory
   occurrence remains in the final runs. Different code requires source review;
   a small diff or a legacy family match is insufficient to copy a fingerprint.
3. Read construction and train/inference entrypoints and follow active helpers.
   Use the source-backed seed or a coordinator-approved example to guide the
   checklist. Fill every component; do not look only at the parent-child diff.
4. Repeat the component loop below in schema order. Collect semantic facts and
   assign each fact one component owner. Do not calculate families by hand.
5. Account for every `source_inventory` statement. Cite active facts or justify
   a permitted exclusion. Do not bulk-mark unknown statements implementation-only.
6. Call `validate_source_review(review, sources, fingerprint, schema)` from
   `experiments.ontology_categorical_review`. Save its receipt only if it passes.
   A validator passing does not replace checking that the source reasoning is true.
7. Before recording completion, reread the working schema revision. If it
   changed, keep the old result, mark it stale, and ask the coordinator for a
   reprojection assignment. Do not claim it validates under the new revision.
8. Write the result, source bundle and any exception card to your packet's output
   directory. Return counts and paths to the main agent. Do not edit its ledger.

### The same loop for every component

1. Read this component's `scope`, `excludes` and category definitions.
2. Find the source computation occupying that role. Record its input, operation,
   output and consumer. Names such as `activation` or `gate` in source are clues,
   not proof of the role.
3. If the role is absent, cite why no relevant reachable path instantiates it.
   Missing source or an unsuccessful text search is not evidence of absence.
4. If present, compare its law and role with the declared category definition.
5. Resolve equivalent observed spellings using `canonicalize_observations`.
   Deduplicate equivalent observations. Do not concatenate distinct labels.
6. If one defined, nonprovisional category matches, record it and its evidence.
   Otherwise create an exception card. Do not invent a fallback value.

Keep this repeated loop simple. Function spelling, parameter size, role and
mechanism are different questions. For example, sigmoid used as a gate belongs
to the gate-transfer role; sigmoid returning probabilities belongs to the
output-link role. The same reasoning applies to normalization, attention,
recurrence, embeddings, readouts, objectives and all other fields.

### Candidate result shape

Write `<candidate_id>.json` plus `<candidate_id>.source.json` in your assigned
result directory. The source file maps relative source paths to exact text.

```json
{
  "candidate_id": "the exact ID from the packet",
  "status": "validated",
  "schema_revision": "the exact working schema revision",
  "source_bundle_path": "the absolute path to the source JSON",
  "fingerprint": {},
  "source_review": {},
  "validation_receipt": {}
}
```

The empty objects illustrate the envelope only: replace them with the complete
vector, complete `role_evidence_v1` review, and actual validator result. For an
unfinished case use `status="needs_evidence"` and `exception_path`; omit the
fingerprint and receipt rather than inserting dummy values. Source-unavailable
or invalid-source cases retain their evidence and remain outside validated
fingerprint counts until resolved. Never give them a fabricated family.

## Unfamiliar cases: use small evidence tasks

Write one exception card containing candidate ID, source digest, component,
exact source excerpt, the unresolved distinction, and the next concrete fact
needed. Put numbers in the excerpt, not in proposed categorical values.

The main agent turns the card into one or more of these tasks:

| Task | Required output |
| --- | --- |
| Resolve one symbol | Its definition/import, call sites and file/line evidence. |
| Trace one quantity | Producer, transformations, final consumer and relevant branch condition. |
| Compare two functions | Side-by-side operations and the first actual behavioral difference, if any. |
| Check one absence claim | Every relevant call path and the evidence making the suspected path inactive. |
| Recover one source | Candidate-ID-matched artifact locations and verified file contents; never regenerate a candidate. |
| Refine one category | One exact mechanism definition, one equivalent example and one distinguishing example. |

Do not ask one worker to redesign the whole campaign schema. The main agent
uses the evidence to make one local definition/role decision, records it in
`SCHEMA_CHANGES.md`, updates its working schema, and dispatches simple
reprojection tasks. Previously reviewed source statements can guide the update,
but every candidate must validate against the new complete schema. Reuse a
decision only when its stated conditions and source evidence match.

Remain on Luna. If a distinction remains unresolved, request a more specific
piece of evidence from a worker and continue independent packets. If the
necessary evidence truly cannot be recovered, report that exact issue without
inventing a label or claiming completion. Do not silently switch to another model.

## Scheduling and status

After the seed review, assign one packet to each available worker and one to
yourself. Assign ready source reviews, source-recovery tasks, or independent
checks whenever a slot frees up. An agent writing a packet result is its sole
writer; a checking agent writes only its separate check output.

Check the first completed candidate from each worker against the actual source
before approving its repetitive batch pattern. Independently check every new
schema decision, disputed role and reused equivalence rule before propagation.
Run the validator on every completed candidate. Do not approve an entire
program-equivalence class from one superficial syntactic match.

The main agent updates the user at least once per minute while working:

`Validated X/Y unique candidates (P%); schema R; A workers active; E evidence tasks pending. Currently: ...`

Use `P = 100 * X / Y`, where `Y` is `queue.json`'s unique-candidate count,
including the canonical seed once, and `X` is the number validated under the
current schema revision. Stale reviews, exceptions, missing source and mere
file scans do not count. Count batches separately. If a schema revision reduces
the validated count, explain the reprojection instead of hiding the decrease.
Once candidate review reaches 100%, still report final metric validation as
pending until it passes. Do not create scheduled automations for these updates.

## Finish and hand back

1. Verify every unique candidate is accounted for and every occurrence in the
   original inventory remains in its chronological run. Shared source review
   never means deleting repeated proposals from trajectories.
2. Revalidate all candidates under the final working schema. Resolve conflicts,
   missing components and stale evidence. Record any genuinely unrecoverable
   source in `UNRESOLVED.md`; do not disguise gaps as zeros or complete output.
3. Build runs from the inventory, materialize the canonical seed once per run,
   and call `output_document(..., schema_override=working_schema, ...)` with
   all required source reviews and bundles. Retain the full schema snapshot.
4. Run the two shared categorical test files plus campaign-specific checks. Use
   a temporary test directory within your own `BASE` so forks do not collide.
5. Save `BASE/final.json` only after all required evidence and metrics validate.
   `BASE/HANDOFF.md` reports counts, schema changes, exact commands/results,
   output paths and any unresolved issues. Shared-code fix proposals are separate.

The fork completes a campaign review and hands back its artifacts. Dashboard
integration and merging campaign working schemas into the shared registry are
coordinated in the parent task after the campaign results are ready.

## Worker prompt template

The main agent fills every bracket before dispatch; do not send placeholders.

```text
Use gpt-5.6-luna. You own exactly [packet path or bounded evidence task].
Campaign: [campaign]. Working schema: [absolute path], revision [revision].
Read the routine worker/component loops in docs/ONTOLOGY_LUNA_FORK_PLAYBOOK.md
and docs/ONTOLOGY_COMPONENT_REVIEW_RULES.md.
Read this source-backed example: [validated seed/result path, or explicit none
for seed preparation].
Write only under [absolute assigned result directory]. Other agents are working.
Do these steps: [short numbered task-specific list].
For an undefined mechanism, return an exception card with exact source evidence;
continue independent candidates. Do not guess, join labels or change schemas.
Run [exact validation function/command] on each claimed complete result.
Return: packet ID; validated candidate IDs; exceptions and exact next evidence
needed; schema revision; output paths; validation results.
Do not create user-visible tasks or delegate further without main-agent scheduling.
```

## Runtime and source references

Run commands from the repository root with
`outputs/tiny-seed-search/venv/Scripts/python.exe`. Preparation, if packets are
missing, is `-m experiments.prepare_ontology_luna_packets --campaign <campaign>`.
Reruns preserve existing packet contents and working-schema edits; inventory
changes require reconciliation rather than overwriting in-progress work.

OpenAI documents the model ID as [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
The bounded task descriptions, explicit model selection and separate write
ownership follow [the official subagent guidance](https://learn.chatgpt.com/docs/agent-configuration/subagents).
Packet size and workflow steps above are project choices based on the operator's
request, not claimed universal Luna performance guarantees.
