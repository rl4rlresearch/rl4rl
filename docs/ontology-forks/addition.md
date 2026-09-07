# Addition: Luna fork instructions

Complete the categorical source review for **Addition only**. Follow
[the Luna fork playbook](../ONTOLOGY_LUNA_FORK_PLAYBOOK.md) and the
[component review rules](../ONTOLOGY_COMPONENT_REVIEW_RULES.md).

Use **gpt-5.6-luna** for this fork and all workers. This replaces the earlier
Terra instruction for this work. **Start as many useful Luna subagents as the
runtime permits, divide the work between yourself and those subagents, and
keep doing your own assigned work while they run.** Reuse workers and refill
available slots. Do not assume the cap is ten or create user-visible tasks.

Give each worker one small packet or one concrete evidence task, with exact
input paths, schema revision, output directory and validation steps. Follow
the playbook's component loop repeatedly. Break unfamiliar cases into symbol
resolution, dataflow tracing, source comparison and one-definition tasks.
Keep the method's correctness checks; do not guess to finish faster.

## Fixed inputs

- Campaign ID: `addition`.
- Source campaign key: `openevolve_v21`.
- Source root: `C:/Users/hpn-w/Documents/GitHub/rl4rl/data/c0c3/controlled-openevolve-transformer-v2-1-mps-campaign`.
- Inventory: `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/inventories/addition.json`.
- Queue: `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/forks/addition/queue.json`.
- Shared seed packet: `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/forks/addition/packets/seed.json`.
- Working schema: `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/forks/addition/working-schema.json`.
- Campaign output root: `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/forks/addition`.
- Initial unique-candidate denominator: **1875**, including the seed once.
  Read the queue for the authoritative count; retain all trajectory occurrences.
- Scope: C0-C3 only, excluding dashboard-configured B03-C0, B03-C3, B04-C2 and B05-C1. Include the shared seed and proposals 1 through 120 inclusive. This leaves 16 runs; ignore later proposals.
- Tiny Adderboard is handled in its own fork, limited to proposals through 100 inclusive.

## Corrected source references

Source recovery and packet scope have been reconciled. Read [the source-resolution rules](../ONTOLOGY_SOURCE_RESOLUTION.md) and the campaign `source-recovery/completion.json` before continuing. Use the shared recovery and rebuild commands there; preserve existing packet IDs and resolved source aliases.

## Start immediately

1. Read the playbook, queue, working schema and existing progress/results.
2. Assign bounded seed tracing, training tracing and coverage tasks to Luna
   workers, up to the current available capacity. You assemble the single
   source-backed seed review. Extra slots can check source locations in
   separate batches.
3. After seed validation, give each worker one numbered candidate packet and
   take a packet yourself. Save after each candidate. Keep slots busy.
4. Only you update `working-schema.json`, `progress.json` and the integration
   outputs. Workers write only inside their assigned packet/check directories.
5. Use precise exception cards for undeclared mechanisms; delegate the next
   small evidence question and continue independent candidates. Extend the
   local schema when necessary and reproject all affected earlier reviews.
6. Report actual validated unique candidates/current denominator and percentage
   at least once per minute. Exceptions and stale reviews are not complete.
7. Integrate every original trajectory occurrence and publish through
   `output_document("addition", ..., schema_override=working_schema, ...)`.
   Write final artifacts to `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/forks/addition/final.json` and
   `C:/Users/hpn-w/Documents/GitHub/rl4rl/outputs/ontology-categorical-v1/forks/addition/HANDOFF.md` only after the required checks pass.
   Record genuinely unresolved source in `UNRESOLVED.md`; do not claim 100%
   classification or fill gaps with dummy categories.

## Commands

Run from `C:/Users/hpn-w/Documents/GitHub/rl4rl`. Packet generation is already done; use this only
if needed, preserving existing progress and schema edits:

```powershell
& outputs\tiny-seed-search\venv\Scripts\python.exe -m experiments.prepare_ontology_luna_packets --campaign addition
```

Run these shared checks plus your campaign-specific checks at final integration:

```powershell
& outputs\tiny-seed-search\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/ontology-categorical-v1/forks/addition/pytest-final tests/test_ontology_categorical_fingerprint.py tests/test_ontology_categorical_review.py
```

Do not edit the shared schema, shared Python modules, shared tests, another
campaign's outputs, original campaign source, or legacy labels. The local
working schema remains extensible. Record any necessary shared-code patch
proposal with a minimal reproducer under this campaign's output root.

