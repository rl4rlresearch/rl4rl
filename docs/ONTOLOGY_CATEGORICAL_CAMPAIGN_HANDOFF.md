# Categorical Ontology Campaign Handoff

Use one fork per campaign. Each fork classifies only its inventory and does not
change the common equality rule or the legacy ontology output.

## Model and parallel execution

Use **Luna (`gpt-5.6-luna`) for each fork and every subagent**, superseding the
earlier Terra preference. **Each fork must start as many useful subagents as
its runtime allows and divide the work between itself and those subagents.**
The main agent also completes assigned work while workers run. Refill free
slots; use the actual capacity rather than assuming the earlier requested ten
workers are available.

Follow the [Luna fork playbook](ONTOLOGY_LUNA_FORK_PLAYBOOK.md). It supplies a
repeatable component loop, ten-candidate work packets, small evidence tasks for
unfamiliar mechanisms, validation instructions and measured progress updates.
Each campaign has an isolated working schema; schema extensions stay allowed.

Start a fork with its ready-to-use instruction file:

- [Addition](ontology-forks/addition.md)
- [nanoGPT](ontology-forks/nanogpt.md)
- [Fashion MNIST](ontology-forks/fashion.md)
- [KWS](ontology-forks/kws.md)
- [UCI HAR](ontology-forks/har.md)
- [Tiny Adderboard](ontology-forks/tiny_adderboard.md)

## Campaign scope

| Campaign | Inventory | Runs | Records | Source recovery items | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| Addition | `outputs/ontology-categorical-v1/inventories/addition.json` | 16 | 1,936 | 0 | Dashboard-configured runs; seed and proposals through 120 |
| nanoGPT | `outputs/ontology-categorical-v1/inventories/nanogpt.json` | 12 | 732 | 0 | All recorded proposals |
| Fashion MNIST | `outputs/ontology-categorical-v1/inventories/fashion.json` | 20 | 4,020 | 0 | All recorded proposals |
| Tiny KWS RNN | `outputs/ontology-categorical-v1/inventories/kws.json` | 20 | 4,020 | 0 | All recorded C0-C3 proposals |
| UCI HAR | `outputs/ontology-categorical-v1/inventories/har.json` | 20 | 4,020 | 68 | All recorded C0-C3 proposals |
| Tiny Adderboard | `outputs/ontology-categorical-v1/inventories/tiny_adderboard.json` | 20 | 2,020 | 9 | Seed and proposals through 100 |

Tiny Adderboard includes only the seed and proposals through 100 inclusive. The source-recovery counts
describe records whose sources have not yet been resolved in the active run folders.
They are explicit recovery work, not unclassified or gray ontology values.
Addition, nanoGPT, Fashion and KWS have had their source references corrected;
see the [source-resolution rules and audit procedure](ONTOLOGY_SOURCE_RESOLUTION.md).
Their former missing IDs often named unsuccessful or duplicate proposal events
whose actual snapshots were already present. Zero remaining source issues does
not mean every proposal passed evaluation.

Each included campaign currently has one source-identical canonical seed across
all its runs. Review that seed once in the inventory's `canonical_seeds` list.
Each proposal-zero record has `canonical_seed_id` and references that one
fingerprint for run-level metrics; do not independently classify seed copies.

KWS also records one seed-only C4 full-search-refresh run per block. C4 is not
part of the C0-C3 experiment and is listed as an excluded run in its inventory.

## Required procedure

1. Read the campaign's initial component registry in
   [`schemas/ontology-categorical-fingerprint-v1.json`](../schemas/ontology-categorical-fingerprint-v1.json)
   and the full method in
   [`ONTOLOGY_CATEGORICAL_FINGERPRINT_V1.md`](ONTOLOGY_CATEGORICAL_FINGERPRINT_V1.md).
2. Follow the mandatory
   [component review rules](ONTOLOGY_COMPONENT_REVIEW_RULES.md). Inspect the
   actual candidate construction, train/inference entrypoints, helpers and
   diffs. Keep excerpts, hashes, numbers and review reasoning outside the
   fingerprint. Do not copy labels from legacy reference catalogs.
3. Give every non-seed candidate, plus each canonical seed source once, a
   complete categorical vector over the current registry. Use `absent` when a
   declared component is not instantiated. Establish computational roles
   before resolving aliases. Do not join distinct functions or select a
   "primary" mechanism when multiple mechanisms occupy one draft role.
4. If a source contains a mechanism outside the registry, add a precise
   category definition or a new component. Do not use `other`, `unknown`, a
   generic custom value, or a partial fingerprint. Edit the campaign's isolated
   `working-schema.json`; supply it through `schema_override` at publication.
   Broad `provisional_values`
   require refinement before publication. New components need scope,
   exclusions, absence rules, numeric invariants and canonical value definitions.
5. After every registry extension, reproject all earlier candidates in that
   campaign to the updated schema, assigning `absent` for a newly introduced
   component where appropriate. Recompute family IDs and all eight metrics from
   proposal zero.
6. Recover every source-recovery item from the campaign data or its preserved
   artifacts. If recovery proves impossible, keep it as a source-unavailable
   provenance record; it cannot receive a fingerprint or family. Do not invent
   a category from the proposal text alone.
7. Generate one output document through
   `experiments.ontology_categorical_fingerprint.output_document`, supplying
   `source_reviews` and `source_bundles` keyed by unique candidate ID and
   `schema_override` from the campaign's working schema. Every
   statement in the source bundle needs a reviewed disposition; every active
   semantic fact has one component owner. One source statement may support
   several independently defined aspects. The output must
   embed the exact current schema snapshot, its content-addressed revision, each
   full fingerprint, family ID, changed-component list, and all eight metrics.

## Completion criteria

The campaign is ready for dashboard integration only when its output has one
schema revision, every fingerprinted candidate validates against that revision,
all direct parents occur earlier in the same run, and no family is based on a
legacy semantic signature or preserving-edge merge. Before handing back,
re-run both categorical fingerprint and categorical review tests. Each new
mechanism needs an equivalence regression (alternate implementation, same
fingerprint) and a distinction regression (changed mechanism, changed
component). Demonstrate role placement and cover unused/helper-code cases.

The exact family definition is unchanged across all six forks: two candidates
share a family if and only if their complete categorical vectors are identical.
