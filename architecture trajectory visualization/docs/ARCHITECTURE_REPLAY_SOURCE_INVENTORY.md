# Architecture replay: source inventory

Inspected on 2026-09-19 from a fresh isolated checkout of
[`rl4rlresearch/rl4rl`](https://github.com/rl4rlresearch/rl4rl), source commit
`65f4aac20963bac4d293879c4e3c2ce96070edfb`. This matches the reference revision
in the implementation brief. The existing local checkout was not used to
define campaign schemas or scientific history. Candidate programs were read
as text; none were imported, traced, trained, or evaluated.

## What the checked-in data actually contains

The tables below describe the **directly tracked event ledgers at this source
revision before any LFS archive restoration**. They are deliberately separate
from published categorical coverage and archive transfer metadata. Publication
counts are not a substitute for the locally available proposal history.

`Completed` counts distinct `proposal_completed` opportunity numbers within each
run. `Incomplete` counts a recorded `proposal_started` opportunity without a
completion. `Last proposal range` is the minimum and maximum last recorded
opportunity across the runs, including incomplete opportunities. All listed
runs also have a seed/run-created record. Multiple lifecycle messages for one
opportunity are not additional replay stops.

| Direct campaign directory under `data/c0c3/` | Runs | JSONL event rows | Completed | Incomplete | Last proposal range |
| --- | ---: | ---: | ---: | ---: | --- |
| `controlled-openevolve-transformer-v2-1-mps-campaign` | 20 | 4,626 | 2,293 | 20 | 34–150 |
| `fashion-mnist-openevolve-v2-1-mps-campaign` | 20 | 8,020 | 4,000 | 0 | 200–200 |
| `nanogpt-openevolve-v2-1-h100-campaign` | 12 | 1,330 | 659 | 0 | 48–59 |
| `tiny-kws-rnn-openevolve-v2-1-cpu-campaign` | 20 | 2,018 | 989 | 20 | 39–61 |
| `tiny-kws-rnn-openevolve-v2-1-cpu-campaign-invalid-launch-20260903` | 20 | 94 | 34 | 6 | 2–2 |

The invalid-launch folder reuses KWS run names and protocol identifiers but has
different run-created timestamps and interrupted launch history. It is a
separate historical revision, not 20 additional comparable KWS experiments and
not a source to merge into the current KWS trajectories. The architecture
catalog omits invalid-launch/quarantine roots from its default comparable
scope; archive scope may expose them with their distinct campaign provenance.

Other `data/c0c3/` directories include calibrations, supervisors, and standalone
copies. A standalone run matching a campaign run is not an additional run.
See [the restoration notes](../../experiments/WINDOWS_DASHBOARD.md).

## Exact comparable dashboard scope

The existing helpers in
[`live_trajectory_dashboard.py`](../../experiments/live_trajectory_dashboard.py)
are the source of the presentation policy:

- Addition excludes run suffixes `b03-c0`, `b03-c3`, `b04-c2`, and `b05-c1` from
  the `controlled-openevolve-transformer-v2-1-source-only-ten-digit-addition-pair-transformer-openevolve-v2-1-mps-openevolve-`
  run family.
- Every run whose ID contains `ten-digit-addition` is capped at proposal 120.
- The corresponding archive view retains extra recorded proposals and the
  excluded runs. Changing the view does not modify underlying artifacts or
  extend reviewed categorical coverage.

Applying those rules to the direct ledgers gives:

| Task | Comparable runs | Completed proposals | Incomplete proposals | Seed + opportunity stops | Last proposal range |
| --- | ---: | ---: | ---: | ---: | --- |
| Addition | 16 | 1,873 | 6 | 1,895 | 108–120 |
| Fashion-MNIST | 20 | 4,000 | 0 | 4,020 | 200–200 |
| nanoGPT | 12 | 659 | 0 | 671 | 48–59 |
| Tiny Keyword Spotting | 20 | 989 | 20 | 1,029 | 39–61 |
| **Total directly available** | **68** | **7,521** | **26** | **7,615** | — |

These event-derived counts were independently reproduced by the architecture
normalizer across all 68 comparable direct runs, including opportunity-result
fallbacks. Its occurrence-level source-available counts, including seeds, were
1,889 Addition, 4,020 Fashion, 671 nanoGPT, and 1,006 KWS. An available stored
source is still not a guarantee of a complete extracted computation graph.
The normalized catalog must label its active scope and data revision.
Restoration of a later archive can legitimately
change counts; it must not silently overwrite a conflicting tracked revision.

## Source coverage and identity hazards

The recorded artifact reference, source identity, and proposal candidate ID
are separate fields. In the direct raw ledgers:

| Current campaign | Completed records absent from `state.candidates` | Repeated candidate-ID occurrences within a run | Artifact directory name differs from candidate ID |
| --- | ---: | ---: | ---: |
| Addition, before presentation exclusions | 1,141 | 15 | 16 |
| Fashion-MNIST | 1,500 | 296 | 268 |
| nanoGPT | 35 | 2 | 10 |
| KWS | 401 | 0 | 11 |

This is why replay reads event history rather than iterating only the final
state's candidate map, and why temporal ancestry uses distinct proposal
occurrences. Resolving `candidates/<candidate_id>` alone loses valid source
aliases and must not define availability.

As a filesystem check, the recorded/rebased artifact directory contained at
least one Python file for 2,293/2,293 Addition completed rows, 4,000/4,000 Fashion
rows, 659/659 nanoGPT rows, and 986/989 KWS rows. **This is presence coverage,
not certified source identity or graph extraction coverage.** An interrupted
attempt may point at a parent fallback; file presence does not establish that
the attempted source was unchanged. Apply the exact editable-file bundle
matching rules in [source resolution](../../docs/ONTOLOGY_SOURCE_RESOLUTION.md).

The seed comes from run creation/manifest evidence even when its objective is
missing. Proposal source joins must retain campaign, run, proposal, recorded
candidate ID, source candidate ID, and source digest. Archive paths originating
on Windows require the existing safe rebasing rules, not arbitrary filesystem
access. Python module order and categorical fingerprints do not prove tensor
connectivity.

## Archive assets and restoration provenance

At initial inspection all three ZIP paths were **Git LFS pointer files**, not
downloaded archives. Their payload identities are:

| Archive in `data/` | Payload bytes | SHA-256 / LFS object ID |
| --- | ---: | --- |
| `rl4rl-c0c3-dashboard-filtered-20260906.zip` | 728,375,453 | `ca0b23195fa75e8272313014838257ee73a60f5dc0603f711861deb3bdc6f0d5` |
| `rl4rl-tiny-adderboard-through-100-c0-c3-20260908.zip` | 92,766,796 | `bd9c54e957fafde51d8edce1512041febac9fe796d13c62bed4694e399a8f592` |
| `rl4rl-uci-har-c0-c3-20260908.zip` | 569,518,648 | `1f1d3ba586eea507df2536f0e329599c1c81fee029bead1ec555949691942bdb` |

`.gitattributes` marks these archives and the five ontology `final.json` files
as LFS assets. Detect the `version https://git-lfs.github.com/spec/v1` header
explicitly. A successful checkout with pointers does not establish that these
datasets are prepared.

The checked-in
[`rl4rl-tiny-har-transfer-20260908.json`](../../data/rl4rl-tiny-har-transfer-20260908.json)
was created at `2026-09-08T06:13:28.671892+00:00` and reports:

| Archive campaign root | Runs | Completed proposals | Proposal limit | ZIP entries | Metadata file checksums | Missing candidate-directory entries |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `tiny-v21` | 20 | 2,000 | 100 per run | 40,068 | 60 | 9 |
| `uci-har-pareto-v21` | 20 | 4,000 | No transfer cap; every run ends at 200 | 78,778 | 60 | 68 |

The metadata records CRC verification, scope verification, and unchanged source
metadata at transfer time. Those are recorded transfer claims, not a new local
verification of unhydrated payloads. Restore tooling must verify the downloaded
payload digest, validate member paths, and preserve conflicting existing
revisions. Missing recorded candidate directories require source-alias checks;
they do not automatically imply missing program evidence.

The direct checkout already supports Addition, Fashion, nanoGPT, and KWS.
Tiny AdderBoard and UCI HAR require their corresponding archive payloads.
The main filtered archive is also relevant to later histories: the published
Addition, nanoGPT, and KWS coverage extends past the direct tracked ledgers.
Do not manufacture those later occurrences from categorical metric rows.

### Verified local preparation

The reproducible preparation command is:

```sh
python "architecture trajectory visualization/run.py" prepare --campaign all --workers 32
```

It obtains public GitHub LFS payloads without a Git LFS installation, resumes
download ranges, verifies each complete archive against its pinned Git blob,
and extracts only source/metadata into separate prepared roots. Existing
different revisions and unidentified destination files are refused. Interrupted
extractions retain an archive identity marker; only a completed verified receipt
makes a prepared root available to the dashboard. Candidate code is never run.

The Tiny archive was subsequently downloaded from GitHub's public LFS endpoint,
verified against its full recorded SHA-256, and restored read-only into the
separate repo-like root `outputs/architecture-replay-data/tiny/`. The preparation
CLI extracted 20,130 source/metadata files. Its normalized catalog contains
20 runs and 2,020 seed/proposal occurrences; stored source resolves for all
2,020 occurrences, including source programs that are syntactically invalid.
The receipt is `architecture-replay-source.json` in that prepared root.

The UCI HAR archive was also downloaded, matched to its complete recorded
SHA-256, and restored into `outputs/architecture-replay-data/har/`. Preparation
extracted 39,738 source/metadata files. All 60 original manifest/state/event
checksums match. The normalized archive contains 20 runs and 4,020
seed/proposal occurrences, including 239 failed evaluations, 3,302 retained
proposals, and 459 rejected proposals. Retention describes the recorded
portfolio decision; it does not establish a new global best. HAR remains
unreviewed by the published categorical ontology. Source identity is established
for 4,019 occurrences. Run
`uci-har-pareto-v21-uci-har-pareto-openevolve-b02-c3`, proposal 171, is an
infrastructure interruption whose recorded parent-artifact reference is not
proof of attempted source. It remains `unresolved_recovery_fallback`, with no
substituted graph or source hash.

The filtered archive was then verified against its complete SHA-256 and
restored into `outputs/architecture-replay-data/filtered/`. Its 235,015 ZIP
members contain four campaigns; preparation extracted 113,376 source/metadata
files. This prepared revision replaces the four older direct campaigns as a
whole in the comparable dashboard. It does not splice together their ledgers.
Archive scope preserves the older raw checkout histories as separately labeled
revision entries, including the excluded Addition runs and extra proposals.

The final active comparable catalog, after all three archives were prepared,
was independently normalized through `ArchitectureStore`:

| Task | Prepared root | Runs | Completed proposals | Seeds + proposal occurrences | Source available | Last proposal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Addition | `filtered` | 16 | 1,920 | 1,936 | 1,936 | 120 |
| Fashion-MNIST | `filtered` | 20 | 4,000 | 4,020 | 4,020 | 200 |
| nanoGPT | `filtered` | 12 | 720 | 732 | 732 | 60 |
| Tiny Keyword Spotting | `filtered` | 20 | 4,000 | 4,020 | 4,020 | 200 |
| Tiny AdderBoard | `tiny` | 20 | 2,000 | 2,020 | 2,020 | 100 |
| UCI HAR | `har` | 20 | 4,000 | 4,020 | 4,019 | 200 |
| **Total active comparable catalog** | — | **108** | **16,640** | **16,748** | **16,747** | — |

There are no incomplete proposal occurrences in these prepared comparable
revisions. Completed counts include failed and invalid evaluations; they do not
mean successful training or retention. All source availability figures use the
same interrupted-workspace identity gate as HTTP snapshots and portable
exports. Source availability does not imply syntactically valid Python or a
complete extracted architecture. In particular, both documented Tiny syntax
exceptions remain available as source but unavailable as reconstructed graphs.

There is an intentional distinction between **archive** and **original source
metadata** checksums. All 20 Tiny run manifests match the transfer manifest's
`source_metadata_sha256`; the 20 state files and 20 event ledgers do not. The
archive's `TRANSFER-README.txt` explains that the capped export filters event,
lifecycle and usage ledgers plus state to proposal 100 while leaving original
source campaigns untouched. Each state also records
`transfer_export.proposal_limit: 100`. Those original-source digests must not be
misrepresented as expected digests for the transformed export. The archive's
verified SHA-256 authenticates the exported bytes, and the receipt preserves
the individual original/export digest comparisons.

## Published categorical evidence

[`publication-manifest.json`](../../outputs/ontology-categorical-v1/publication-manifest.json)
was created at `2026-09-07T19:36:41.476841+00:00`. It declares:

| Publication | Runs | Classified candidate IDs | Reviewed seed/proposal rows | Unresolved rows | Compact metric rows | Schema revision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Addition | 16 | 1,875 | 1,936 | 0 | 1,936 | `schema_e2cf16ddcaa7c6da` |
| Fashion-MNIST | 20 | 3,705 | 4,020 | 0 | 4,020 | `schema_8178e16b182d2a51` |
| KWS | 20 | 3,940 | 4,020 | 0 | 4,020 | `schema_850adcdf999b8530` |
| nanoGPT | 12 | 702 | 732 | 0 | 732 | `schema_b6aec173a4ddb6d5` |
| Tiny AdderBoard | 20 | 1,985 | 2,018 | 2 | 2,020 | `schema_17d52eae155c5aca` |

All ten directly tracked `working-schema.json` and `trajectory-metrics.json`
files matched their publication-manifest SHA-256 checksums during inventory.
All five `final.json` files were initially LFS pointers. Their total declared
payload is approximately 992 MB; they contain large review bodies and should
not be sent to the browser on each selection.

The compact views contain per-occurrence candidate ID, proposal, fingerprint,
retention, metrics, and missing-source policy flags. They are useful bounded
annotations, not computation graphs or independent source certificates. Join
only compatible campaign/run/proposal/candidate/schema records and report
unverified source provenance honestly when full certificates are unavailable.

UCI HAR is explicitly excluded from the categorical publication because source
review was unfinished. Its trajectories can still be displayed with that
qualification. Tiny AdderBoard B02-C2 proposal 92 and B05-C2 proposal 98 are
syntax-invalid exceptions with no reviewed fingerprint/family. Their compact
`no_change_assumed` policy carries metric totals forward; it does not prove
architectural equivalence. Preserve the full-run novelty basis when users
change the selected path, outcome filter, or comparison.

## Existing components to reuse

- `experiments/live_trajectory_dashboard.py`: campaign configuration, event
  normalization, task metric metadata, run exclusions/cap, read-only source
  endpoint, page routing and cache/revision integration. Its `/api/data`
  completed points need supplemental incomplete-opportunity handling for
  complete replay.
- `experiments/ontology_categorical_inventory.py`: recorded artifact/source
  resolution and interrupted-workspace identity evidence. Its scoped inventory
  is not a replacement for an unrestricted archive timeline.
- `experiments/ontology_categorical_dashboard.py`: primary-parent and full-run
  categorical metric semantics. Preserve published values rather than
  recomputing novelty on the displayed path.
- `experiments/ontology_dashboard.js` and source-review utilities: existing
  static evidence inspection. Reuse source evidence without executing models
  or rewriting reviewed ontology categories.

The separate
`experiments/research_process_interventions_andy/dashboard/app/data/trajectories.json`
declares 24 runs and 320 proposals. Its recorded non-null `publicAccuracy`
values are all `0.0`; it is the older smoke-evaluation IR sweep, not the C0–C3
campaign collection. Its `nodeKinds` is a count summary, and frozen IR snapshots
must be joined to the actual seed/candidate identity before rendering. It must
not substitute for the six-task campaign integration or be presented as
evidence of successful task learning.
