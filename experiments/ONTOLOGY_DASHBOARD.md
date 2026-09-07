# Ontology explorer

Open `/ontology` on the existing dashboard server. This page was duplicated
from `live_trajectory_dashboard.html`; `ontology_dashboard.js` extends that
copy's campaign selection, filters, Chart.js rendering and refresh lifecycle.
The original explorer is unchanged apart from navigation.

The explorer reads candidate IDs, actual parent IDs, portfolio snapshots and
evaluation outcomes from the same saved event data. It never changes a campaign,
runs an evaluator, or treats intervention wording as an ontology annotation.

## Fingerprint coverage

Every task has shared components for input units/transforms, embeddings,
position, mixing, routing, memory/state, feature transformations, weight
construction and sharing, normalization, connectivity, aggregation, output,
symmetry, conditional computation, activation, stochasticity, bottlenecks and
iterative computation. Task extensions cover:

| Task | Additional components |
|---|---|
| Four- and ten-digit addition | Operand representation, digit order, carry representation, attention scoring, projection relationships, output factorization |
| nanoGPT | Context topology, attention scoring, KV memory, projection relationships, vocabulary representation, block composition |
| Fashion-MNIST | Spatial units/operator, scale representation, spatial readout, channel interaction |
| Keyword spotting | Acoustic representation, state update/structure, temporal scheduling, history readout, exit policy |
| UCI-HAR | Sensor fusion, temporal operator, directionality, state update, frequency representation, temporal readout |

Training and inference have independent dictionaries. Numeric widths, ranks,
depths, rates and thresholds go in `settings`. No fixed ontology can enumerate
every future invention: each schema has `other`, and imported fingerprints
can add named components that automatically become timeline rows.

These components are descriptive axes, not automatic boundary judgments.
Ordinary tying, layer deletion and within-family substitutions can change a
fingerprint while remaining ontology-preserving. Reviewers explicitly assign
`changing`, `preserving`, `mixed`, `uncertain` or `unannotated` under a named
rubric. A gating change may be a component-level crossing without replacing
the model family. The family label should use the rubric's chosen granularity.

## Review data

Download the campaign-specific JSON template from the tab. Each row is bound
to a run, proposal opportunity and candidate ID (including seeds). Enter:

- `fingerprint`: categorical architecture mechanisms; null means unreviewed,
  while the string `absent` means reviewed and not present.
- `training`, `inference`, `settings`: independent procedure/settings records.
- `family`, `classification`, `reviewer`, `notes`: stable family label, explicit
  boundary judgment and supporting source evidence.
- `proposed_change`, `implemented`, `executable`: boolean or null. These are
  separate judgments; a failed patch can propose a change without implementing it.

Reviewed files in `outputs/ontology/<campaign-key>.json` load automatically via
the read-only `/api/ontology/reviews` endpoint. These diagnostics are separate
from campaign directories and are never exposed to subject agents. Saved review
rows bind to exact run/proposal/candidate IDs; rows no longer present in the
dashboard snapshot are omitted, and later proposals remain unannotated.

Manual imports override the saved review for the current browser tab. Campaign/
candidate mismatches and duplicate rows are rejected. Importing never writes
to the server; export manual changes before reloading. A reload loads the saved
server review again. Clearing reviews suppresses automatic reimport for that tab
until a page reload.

## Source review now provided

`review_ontology_sources.py` reads a captured dashboard snapshot and candidate
Python artifacts. It parses ASTs without importing candidate code, follows the
selected model factory into referenced classes/helpers, and extracts only
supported component categories. Missing and unresolved components stay null.
Several factory roots are unresolved rather than pooled with inactive alternatives.
Each extracted component has file, line and source-expression evidence. Agent
comments, condition labels and scores do not establish architectural categories.

The corpus-specific rules were checked against all six seeds and examples of
unordered-pair embeddings, bilinear digit composition, fixed-basis coordinates,
single-gate recurrence, candidate-detail GRUs, scalar mixed pooling, multiscale
convolutions and sensor-energy envelopes. Declared module presence does not prove
runtime use: these are partial static fingerprints, not exhaustive manual or
blinded mechanistic adjudication. Candidate evidence states this limitation.

`adjudicate_ontology_reviews.py` makes a second source pass. It compares rooted
execution shapes, allowing standard affine/table/normalization coordinate
wrappers while preserving model-factory choices, input encoding and referenced
global configuration. Unchanged shapes support preserving labels. Unresolved
source changes remain uncertain; unexplained changes are never automatically
called preserving. Pure removal of one already-present positional mechanism is
not counted as introduction of a new primitive.

Reproduce the saved snapshot diagnostic with:

```powershell
python -m experiments.review_ontology_sources
python -m experiments.adjudicate_ontology_reviews
```

Inputs, timestamped outputs and the summary are under `outputs/ontology` and are
not committed as source code. The adjudicated files record hashes of the review
scripts. This is an exploratory diagnostic; campaign-level condition comparisons
should still use a frozen rubric and independently validated boundary labels.

Build the rubric from pooled campaign examples, then freeze it before comparing
conditions. For blinded annotation, remove condition/treatment/outcome context
in the review workflow; this exploratory dashboard itself is not blinded.

## Chart semantics

- Discovery counts unique reviewed families, excluding a reviewed seed family.
  Unannotated runs have gaps, not invented zero discoveries. Partial review
  produces a lower bound. Stage selection uses implemented/qualified/retained
  evidence and counts full history before applying the visible window.
- Outcome bars count yes/no/unknown independently, not a falsely nested funnel.
  A nonqualification evaluation provides evidence of execution; an unmatched
  patch does not. The cohort selector restricts to reviewed crossings or
  preserving edits. The qualification bar uses the campaign's saved validity
  outcome, including structural validity for tasks without a performance gate.
- Timeline borders compare known categories with actual parents, never the
  previous chronological proposal. These borders are categorical differences,
  not automatic ontology classifications. Missing parent categories remain unknown.
- Lineage uses recorded parents, with branches separated vertically. Y has no
  numerical meaning. Missing ancestors are left disconnected. Orange edges mark
  reviewed changing/mixed edits. Retained nodes have a light border.
- Portfolio bars count slots by family. Missing snapshots are gaps. Unknown
  members remain an explicit unannotated family.
- Persistence examines reviewed implemented crossings with one known parent
  and at least one changed, known component. At each descendant depth, a
  descendant retains the mechanism if all those component categories match.
  Denominators and unknown counts are displayed. This is a descriptive count of
  crossing-descendant pairs, not a survival estimator or independent sample size.
- Performance lines retain the original run colors; individual markers use
  ontology colors and triangles for reviewed crossings. Clicks open candidate
  evidence. Aggregated/intervention performance views retain original semantics.

Family discovery has compact local condition toggles, a Runs / Condition mean selector,
and a stage selector inside its panel. Its condition selection is independent of
the main legend; individually hidden runs and proposal bounds still apply. Means
align by proposal and require known values from every included run in the condition;
missing proposals or unknown values produce gaps. Selections survive auto-refresh
and campaign switches for the current page session.

Other ontology panels share condition/run visibility and proposal bounds. The original
performance chart additionally supports its original outcome/type and aggregation
filters. Combined cross-task campaigns are omitted because their ontology
categories are not interchangeable.

Source inspection reads only recorded candidate/parent Python files within the
run's candidate directory. It supports restored Windows separators and shows
missing-source warnings. It never executes candidate code.

The trajectory family key separates run/condition line colors from family marker
colors. Families use rose, yellow and cyan shades consistently across trajectory,
lineage and portfolio charts. Campaign-local F IDs and full family names appear
in the legend and trajectory tooltips; gray means no assigned family. The key
switches to an explanatory message in summary/intervention views.

## Source-program review v2

The v2 reviewer adds configuration specialization, comparisons against recorded
parents, explicit treatment of syntactically invalid proposals, and complete
reference fingerprints for the six exact archived seeds. It never imports or
executes candidate code. Literal configuration evaluation uses a whitelist.
Input-dependent branches, index selections, algebraic weight generators and
unrecognized computation remain visible to the comparison.

`ontology_semantics.py` compares rooted programs under documented dimension,
bias and ordinary affine-coordinate normalizations. `ontology_transition_review.py`
recognizes numeric-setting edits without discarding operation, branch, sign or
index changes. `ontology_affine_readout.py` separately handles independent affine
readout coordinates while retaining which state/history features reach the head.
These are bounded static rules, not a proof of arbitrary Python equivalence or
a universal definition of ontology. Unmatched differences are not automatically
ontology-changing: refactoring, layer counts, training/inference procedures and
new mechanisms can all change source structure.

A transition label and a complete fingerprint are separate results. Complete
fingerprints originate in source-hash-locked seed reviews. They propagate only
through exact-source or settings-normalized matches. Preserving edits can change
sharing or parameter-construction details, so other preserving matches inherit
family membership without automatically inheriting every component value.

The dashboard reports assessed records, resolved transitions, complete
fingerprints, uncertain cases and invalid source separately. The static v2 workflow
used invalid source for AST parse failures corroborated by campaign preflight.
The continuous service also supports exact-source undefined-constructor findings
with independent execution-failure evidence, as described below. No missing
implementation receives an invented architecture. The static v2 workflow only covers its captured snapshot. The continuous
review service below detects and reviews later proposals.

For a fresh run, execute from the repository root with a captured `/api/data`
snapshot. Keep the source rules unchanged during the entire pass:

```powershell
python -m experiments.review_ontology_v2 --snapshot outputs/ontology/v2/dashboard-snapshot.json
python -m experiments.adjudicate_ontology_v2
python -m experiments.apply_ontology_references --snapshot outputs/ontology/v2/dashboard-snapshot.json
python -m experiments.publish_ontology_reviews
```

The last command validates without publishing. Add `--publish` to import the
validated diagnostic into the dashboard; it backs up existing review files and
writes each campaign atomically. Add `--require-complete` to reject publication
if any transition or applicable fingerprint remains unresolved. A successful
review process does not imply that this stricter completion check passes.

`v2/final/audit.json` records actual coverage and hashes; campaign review queues
identify unresolved transitions and fingerprint components. The September 6
initial snapshot contains 17,800 records including seeds; the subsequent captured
cutoff adds 604 records. `dashboard-snapshot-latest.json` is the 18,404-record
cutoff, with the additions retained separately in `dashboard-snapshot-increment.json`.
Pass the latest snapshot explicitly to validation/publication. Its adjudication rules are
archived under `outputs/ontology/v2/engines/5bdd1fc2a06a`. The frozen launcher
reuses a parsed program across checks; equality with the uncached computation
was checked on 18 programs spanning all six campaigns. The initial extraction
was exploratory while rules were being developed; the audit hashes its actual
saved outputs as inputs, and the final adjudication recomputes matches with the
frozen rules. Do not describe this as a blinded or independently accuracy-scored
corpus annotation.

`ontology_reference_reviews.json` contains thirteen additional direct source
reviews: proposals 1–12 in the first ten-digit addition run and nanoGPT B02-C0
proposal 4 (compiler autotuning only). The full addition seed model
and every source diff in that set were read. Their ordinary bias, tying and
LayerNorm-coordinate edits are preserving under the handoff's rubric, while the
fingerprints retain changes to parameter construction and sharing. The separate
reference-application command checks both candidate and recorded-parent source
hashes before applying transition labels; identical sources elsewhere can receive
the complete fingerprint without inheriting an unrelated transition label.
Conflicting full fingerprints never propagate through a shared settings hash.
These thirteen direct reviews do not constitute a manual review of the whole corpus.


## Continuous task-specific review

Run the watcher from the repository root using the repository Python environment:

```powershell
outputs/tiny-seed-search/venv/Scripts/python.exe -u -m experiments.ontology_review_service --watch --interval 60
```

Each pass captures the live dashboard campaign inventory, reads candidate and
recorded-parent Python source, and freezes the reviewer code and reference data
under `outputs/ontology/live/engines/<version>`. A subprocess uses that immutable
version for the entire pass. Candidate code is never imported or executed. A
process-held lock prevents overlapping writers and releases after a crash. Source
inventories, modification times and file sizes invalidate the source cache;
source hashes and the reviewer version key the comparison cache. New helper
files and changed parent sources trigger fresh review.

Task modules distinguish ordinary parameter/settings edits from changes to the
represented computation, state, routing, feature statistics and parameter
construction. Unsupported code enters a durable queue. Neither a completed pass
nor a successful parse implies a semantic classification. A positive mechanism
witness can resolve a transition while other fingerprint components still need
review; those two coverage counts remain separate.

`outputs/ontology/live/status.json` records the watcher PID and last phase.
Check that the process is actually alive before treating it as a running watcher.
`live/audit.json` records the current snapshot and both completeness gates;
`live/queue/<campaign>.json` records outstanding source cases. Prior published
reviews are retained by content hash in `live/prior-reviews`. Publication validates
coverage and categorical values before atomically replacing each campaign file.
Each validated campaign is published immediately, independently of campaigns
still being reviewed. The audit's `pending_campaigns` and `pending_inventory`
keep both completion gates false until the captured inventory is fully handled.
`live/revoked-processors.json` can withhold a task reviewer by its processor hash
without blocking unaffected tasks; revocation is checked before review and again
before publication. `live/revoked-engines.json` withholds an entire engine when
the defect affects shared review infrastructure.
A recurring Codex follow-up checks watcher health and investigates newly
unsupported semantic cases. Its existence alone does not establish complete
coverage; the audit and source evidence do.

The open ontology tab checks saved review revisions during its regular refresh.
Changed campaign point inventories force a full reload even if the review file
version did not change. Manual review imports and clearing the local review take
precedence for that page session; reloading restores automatic server review
updates. Completion means every captured proposal has a supported transition
classification and every applicable architecture has a complete fingerprint.

The task-specific reviewers build complete candidate fingerprints from the
reachable model and inference program, including source-reviewed custom model
references. A source-bound candidate profile can be reused for identical source;
its transition label still requires the recorded parent comparison. Preserving
comparisons can reconcile different descriptions of the same family, but never
fill missing fingerprint components. Family identifiers derive from semantic
signatures. A changing witness inside an established preserving-equivalence
family, or conflicting fingerprints for identical candidate source, enters the
review queue and fails the completion gate.

The handoff's preserving examples include weight tying, attention-projection
simplification, ordinary affine coordinates, normalization, and layer/width
changes. Fingerprints retain these details without necessarily creating a new
family. New state or feature primitives, input-dependent routing, positional
representation changes, and nonlinear/algebraic parameter generators require
explicit source witnesses. Fixed inference weights and calibration remain
settings; changing relative view weights based on the input introduces routing.
Here, fixed view pooling includes arithmetic, geometric and power means and a
uniform pointwise probability transformation with fixed coefficients. These
remain inference settings, like temperature calibration. Routing boundaries use
additional cross-class confidence/agreement descriptors to choose relative view
weights, select a computation, or apply hard-decision corrections. Rewriting a
fixed polynomial probability map as a product with the probability does not
create a different family.

A syntactically valid proposal can have a complete fingerprint for its attempted
architecture even when execution fails. Such a review keeps its transition label
and records `executable: false` plus the source-supported execution diagnostic.
For example, a new fast/slow recurrence can be fully described while a declared
gate width mismatches its caller. `invalid_source` means the submission does not
specify an implementable architecture: either Python parsing fails with campaign
preflight corroboration, or an exact-source manual review identifies an undefined
reached constructor and the campaign independently records an execution failure.
The latter requires the precise call location, missing-symbol inventory, source
hash and absence of imports or dynamic name construction that could resolve it.
A similarly named class is never substituted for the missing definition. These
rows carry `invalid_kind`, no family, and a fingerprint marked not applicable.
