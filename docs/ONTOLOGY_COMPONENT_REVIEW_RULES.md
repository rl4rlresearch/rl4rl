# Rules for reviewing categorical components

These rules apply to every component in every included campaign. The family
rule stays simple: a family is an exact complete categorical vector. Semantic
decisions happen when defining and assigning components, before family IDs or
metrics are computed. There are no later exceptions merging unequal vectors.

## The review unit

Classify a mechanism **in its computational role**, traced from the actual
candidate entrypoints. An API name does not establish that role. Read
construction, calls, data dependencies, train/evaluation switches, helpers and
the consumer of each computed quantity. Do not infer absence from a failed
regex match or from source code not being available.

Every component has a scope and excludes neighboring roles. Every allowed
value has a definition. An undeclared mechanism, unresolved role, or broad
draft category requires refinement of the living schema. Some initial category
names remain explicitly provisional; they cannot pass publication checks.
The [campaign role catalog](ONTOLOGY_COMPONENT_ROLE_CATALOG.md) lists the
scope and exclusions of every current component.

| Situation | Required treatment |
| --- | --- |
| Module/functional APIs implement the same law in the same role | Resolve through the role's explicit alias map to one category. |
| Aliased imports, renamed variables, reordered independent declarations | Follow their bindings/dataflow; preserve the fingerprint if the mechanisms are unchanged. |
| The same mechanism occurs repeatedly in a role | Deduplicate its categorical observations; keep locations/counts in evidence. |
| A function is used in different roles | Classify each role separately, even if the API spelling is identical. |
| Distinct mechanisms coexist inside one draft role | Split the role by semantic location, or define an explicit structural composition when the composition itself is the mechanism. Reproject the whole campaign. |
| Source spells out an operation that another implementation fuses | Establish the same law, then use the same value. Matching the class name is insufficient. |
| Unused code or an inactive branch | Record why it cannot affect any relevant path. Do not create a component value from its tokens. |
| Training-only mechanism | Record it in the appropriate training role; it is not dead code merely because inference omits it. |
| Numeric tuning only | Retain numbers in evidence, with the same categorical value. Do not spell out numbers in labels. |
| A number selects a different executed mechanism | Inspect that mechanism explicitly. Do not indiscriminately strip numbers that determine branches, axes or operator identity. |
| One expression implements several semantic aspects | Each aspect has an explicit component owner and justification. The same aspect is not copied into multiple fields. |

Joining tokens with `+`, lowercasing a prose label, choosing the dominant
function, hashing normalized code, and trusting a legacy "complete" flag do not
establish a categorical fingerprint. Alias normalization only applies after
roles and operator semantics are reviewed. For example, axis selection,
normalization statistics and recurrent reset placement can invalidate an
apparent API-level equivalence.

## Examples across components

| Source behavior | Components |
| --- | --- |
| `nn.SiLU` and `F.silu` both transform backbone features | `activation=silu` once. |
| SiLU backbone plus a sigmoid multiplier derived from a spatial channel summary | `activation=silu`, `feature_modulation=multiplicative_gate`, `modulation_source=global_channel_summary`, `modulation_transfer=sigmoid`. |
| That gate has a SiLU hidden control layer | Also `gate_hidden_activation=silu`; its final sigmoid remains the transfer. |
| Sigmoid converts class scores to probabilities | `output_link=sigmoid`; it is not a feature gate. |
| Softmax normalizes attention weights | `attention_weighting=softmax`; it is not an output probability link. |
| LayerNorm module versus equivalent functional layer normalization | One `layer_normalization` value in the reviewed normalization role. |
| Depthwise versus dense convolution | The operation remains `learned_convolution`; `spatial_connectivity` records the channel relation. |
| A readout uses tied embedding weights | Readout map and the across-subsystem parameter tie have separate owners. |
| Fixed view weights change from equal to unequal constants | `view_weight_policy=fixed` in both; coefficient equality creates no category. |
| A probability mixture uses stable log-domain arithmetic | Same mixture mechanism, with its output link recorded separately. |
| Native recurrent cell versus a manual implementation | Check the update equations, reset placement and state roles before declaring equivalence. |
| GRU's internal sigmoid/tanh | Owned by `state_update`; do not additionally report an ordinary backbone activation or extra feature gate. |

These examples are not a list of all possible architectures. New combinations
must extend the schema if their roles or mechanisms are not represented.
Component count is a count of changed declared aspects, not a count of code
edits or statistically independent innovations. Distinct aspects of one
architectural intervention may change together.

## Source review record and publication

Use `experiments.ontology_categorical_review.source_inventory(sources)` to
enumerate statements in the complete supplied Python source bundle. No
candidate code is executed. Source evidence includes helpers and inactive code
so neither can silently disappear from review.

Each unique candidate has one `role_evidence_v1` record containing:

- The current `schema_revision`, `source_sha256`, reviewer and entrypoint analysis.
- A `facts` ledger. Each fact has an ID, one owning component, a semantic
  `aspect`, source statement IDs, observed mechanisms and `role_reason`.
- A `components` entry for every declared role, listing its facts and reasoning.
  An absent role has no active facts and an explicit absence explanation.
- `statement_coverage` for every enumerated statement. Each statement either
  cites its mechanism facts or explains an exclusion as `unreachable`,
  `numerical_setting`, `implementation_only`, or `covered_by_composite`.
  Composite coverage must reference the owning fact; unresolved coverage fails.

The outer declaration of a function may be covered by its internal mechanism
facts. Imports and bookkeeping may be implementation-only. An active operation
must not be dismissed as implementation-only merely because it lacks a schema
slot. A composite exclusion must establish that no additional independent
semantic aspect is being omitted.

Call `output_document(campaign_id, runs, canonical_seed_fingerprints=...,
source_reviews=..., source_bundles=...)`. The review and bundle dictionaries
are keyed by unique candidate ID, so each campaign's shared seed is reviewed
once. The publisher verifies source/schema binding, exact component coverage,
fact ownership, statement coverage, consistent shared candidates, and declared
role dependencies. It retains the review records and audit receipts with the
metrics. Direct low-level metric calculation does not certify a review.

Both the schema and the candidate review must be updated after a new role or
category is introduced. All candidates are reprojected and all eight metrics
are recomputed under the same resulting revision. Source hashes, proof notes
and statement IDs never become family components.

## Regression and review obligations

For every new mechanism or component, retain a positive equivalence pair and a
negative distinction pair. Positive pairs cover module/functional aliases,
equivalent fused/expanded computation, numeric tuning and irrelevant dead-code
changes where applicable. Negative pairs change the actual computation or
move it to a different role. They must change the intended component and leave
unrelated roles unchanged. A single canonical-output fixture does not show
that a classifier understands the distinction.

Run the categorical fingerprint and review tests and
`python -m experiments.audit_ontology_categorical_roles`. The audit scans every
legacy fingerprint field in the active inventories, flagging capitalization,
spacing and joined-label issues as source-review leads. A joined label is not
automatically wrong: source inspection determines whether it is duplicate
spelling, different roles, or a real composition needing a schema extension.

The automated gate proves structural integrity of the supplied review evidence.
It cannot prove arbitrary program equivalence, that a supplied source bundle
omits no external code, or that a reviewer's semantic explanation is true.
Reviewers must verify bundles against candidate artifacts and challenge
exclusion/ownership decisions. Do not report legacy candidates as reclassified
merely because these checks were added.
