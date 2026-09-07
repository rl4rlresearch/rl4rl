# Categorical Ontology Fingerprint v1

This is the post-campaign diagnostic for four included C0-C3 tasks: ten-digit
Addition, nanoGPT, Fashion MNIST, and Tiny Keyword Spotting RNN. Tiny
Adderboard and UCI HAR are excluded. Addition includes the seed and proposals
through proposal 120; later Addition proposals are outside this analysis scope.
Only C0, C1, C2, and C3 runs enter the diagnostic. Seed-only KWS C4 refresh
runs are outside this scope.

The diagnostic separates source evidence from the ontology measurement. Source
code, diffs, scalar hyperparameters, numerical constants, scores, and file
hashes are retained as evidence and metadata. They are never fingerprint values.

## Fingerprint and family

For a campaign schema with components `c_1` through `c_k`, each candidate has
one complete fingerprint:

`F(candidate) = {c_1: category_1, ..., c_k: category_k}`

Every value is one lower-case categorical label containing letters and
underscores only. Each schema component permits `absent`; this means the
component is not instantiated in that candidate. A fingerprint has every
component exactly once. It cannot contain an integer, float, numeric string,
free-text explanation, source hash, or partial value.

A family is exactly one unique complete vector `F`. Two candidates belong to
the same family if and only if every component value matches. The family ID is a
hash of canonical JSON for that vector, but the vector itself is retained for
inspection. There is no semantic-edge merge, code-normalization merge, or
case-by-case exception after fingerprinting.

The campaign component registries are in
[`schemas/ontology-categorical-fingerprint-v1.json`](../schemas/ontology-categorical-fingerprint-v1.json).
They start the source review with roles for nonnumeric mechanisms in each candidate:
input and data representation, feature construction, information routing,
state, readout, conditional computation, stochastic runtime behavior,
inference protocol, learning objective, optimizer family, and categorical data
transforms. Task-specific components add mechanisms such as carry
representation, multi-view fusion, and acoustic exits. Coverage is established
from each candidate's source; the initial list is not an exhaustiveness claim.
An uncovered mechanism requires a schema extension before publication.

The mandatory classification procedure is in
[`ONTOLOGY_COMPONENT_REVIEW_RULES.md`](ONTOLOGY_COMPONENT_REVIEW_RULES.md).
Every component now declares its scope, excluded neighboring roles, absence
rule, numeric invariants, canonical definitions, and role-scoped aliases.
API spellings and inventories of function names are source evidence, not
component values. Broad draft categories marked `provisional_values` cannot
be used in published fingerprints until their missing distinctions are defined.

## Living campaign schemas

The equality rule is fixed. Campaign schemas are deliberately living
registries, because a source review can reveal a component or categorical
mechanism that was not known at setup.

An undeclared mechanism is never placed in an `other`, `unknown`, or generic
custom category. It creates a schema-extension request. The reviewer adds a
precisely named component value and definition, or adds a new component. A new
component is assigned `absent` for each prior candidate that does not implement
it. The whole campaign is then reclassified under the extended registry and all
families and metrics are recomputed from the beginning.

Every published result embeds a full schema snapshot and its content-addressed
revision. Rows classified under different schema revisions are not combined in
one family count or trajectory metric. This allows the vocabulary to grow
without making family equality ambiguous.

## Parent and time order

Metrics are calculated separately within each chronological campaign run.
The proposal seed is proposal zero. A non-seed proposal uses the first ID in its
recorded `parent_ids` list as its primary parent. All parents remain provenance;
the first entry supplies the one reproducible parent needed for a scalar
marginal metric. A parent must occur earlier in the same run.

Every included run in each current campaign uses the same seed source. That
source receives one canonical complete fingerprint per campaign. The
proposal-zero record in every run references that single seed annotation and
the output materializes it only to establish each run's baseline metrics. It is
not reviewed once per seed occurrence.

Novelty is evaluated against all earlier proposals in that run, including the
seed and rejected proposals. It is not restricted to the direct ancestor path.

## Eight metrics

For proposal `t`, parent `p(t)`, component `j`, and chronological prior set
`H_t`, the dashboard exposes these y-axis choices:

| Metric | Marginal value at `t` | Cumulative value |
| --- | --- | --- |
| Component edits | Number of components where `F_t[j] != F_p(t)[j]` | Sum of marginal component edits |
| New component states | Among changed components, number whose new value has not appeared for that component in `H_t` | Sum of marginal new component states |
| Family switches | One if `F_t != F_p(t)`, otherwise zero | Sum of marginal family switches |
| New families | One if `F_t != F_p(t)` and the complete vector `F_t` is absent from `H_t`, otherwise zero | Sum of marginal new families |

The seed has zero for all eight metrics. If a component changes from one
non-baseline state back to an earlier non-baseline state, it counts as a
component edit but as zero new component states. If an entire family reappears,
it counts as a family switch but as zero new families. Neither cumulative
novelty series ever decreases.

## Output contract

The shared implementation is
[`experiments/ontology_categorical_fingerprint.py`](../experiments/ontology_categorical_fingerprint.py).
It rejects partial fingerprints, numerical fingerprint values, undeclared
categories, missing components, and parents that are not earlier in the run.
It produces a separate output document with the fingerprint version, schema
snapshot, schema revision, family vector and ID, changed component list, and
the eight metrics.

An inventory's `canonical_seeds` list identifies shared seed sources. Pass one
fingerprint per ID through `output_document(...,
canonical_seed_fingerprints={seed_id: fingerprint})`; all proposal-zero rows
that reference that ID receive the same reviewed vector.

Inventories are built with
`python -m experiments.ontology_categorical_inventory --campaign <campaign>`.
They are written under `outputs/ontology-categorical-v1/` and contain source
paths and lineage only. They intentionally contain no legacy semantic family
labels and no categorical annotations; each campaign fork supplies those
annotations from source evidence.

`output_document` additionally requires `source_reviews` and `source_bundles`,
each keyed by unique candidate ID. Each seed review is supplied once. The gate
checks the current schema and source digests, every component's evidence,
statement coverage, semantic fact ownership, and role dependencies before
calculating publishable metrics. `annotate_run` remains a low-level calculation
utility for tests and analysis; it does not certify a source classification.

The live ontology page currently loads legacy reviews from `outputs/ontology`.
It labels those results as legacy. The new categorical method requires source
reclassification and subsequent dashboard integration; renaming old labels
does not complete that work.
