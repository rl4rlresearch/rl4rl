# Architecture replay verification

Verified on 2026-09-19 in the isolated checkout based on GitHub commit
`65f4aac20963bac4d293879c4e3c2ce96070edfb`. This report covers data normalization,
source identity, HTTP facade contracts, and static review of the frontend data
flow. The final section records browser rendering and interaction verification.
Candidate Python programs were read as text; none were imported or executed.

The final prepared inventory is **108 runs and 16,748 occurrences** across all
six task families. Earlier sections below retain measurements taken before the
filtered archive finished restoring; the final prepared catalog and audit
sections identify the completed data revision explicitly.

## Active data at the time of measurement

Addition, Fashion-MNIST, nanoGPT, and KWS used their directly tracked campaign
histories. Tiny AdderBoard and UCI HAR used the independently restored,
SHA-256-verified GitHub LFS archives under
`outputs/architecture-replay-data/{tiny,har}`. The larger filtered archive was
still downloading. Its later installation may change the active histories;
the numbers below identify the measured revisions rather than claiming the
tracked histories are the latest published histories.

The catalog exposed all six tasks and 108 comparable runs: Addition 16,
Fashion-MNIST 20, nanoGPT 12, KWS 20, Tiny 20, and HAR 20. Archive scope also
preserves invalid-launch history and, when a verified prepared campaign
replaces a tracked campaign, exposes the raw checkout under a separate
`__tracked_archive` key. Different revisions are never combined into one run.

## Selected-frame measurements across six tasks

For each campaign, the first sorted run was B01-C0. The check loaded the
normalized run, its seed, last recorded opportunity, last completed opportunity
when different, and final recorded incumbent when different. Every snapshot
request supplied the run revision. Assertions checked occurrence ID, candidate
ID, source hash, and returned run revision against the selected occurrence.

| Task | Run stops | Seed nodes / edges | Last opportunity nodes / edges | Final incumbent nodes / edges | Run load | Snapshot range |
| --- | ---: | --- | --- | --- | ---: | ---: |
| Addition | 121 | 42 / 35 | #120: 141 / 110 | #117: 140 / 109 | 246 ms | 42–53 ms |
| Fashion-MNIST | 201 | 13 / 20 | #200: 23 / 38 | #192: 23 / 38 | 271 ms | 28–35 ms |
| nanoGPT | 56 | 34 / 17 | #55: 34 / 17 | #52: 34 / 17 | 72 ms | 14–16 ms |
| KWS | 58 | 9 / 5 | #57: unavailable, incomplete | #56: 9 / 5 | 102 ms | 9–11 ms |
| Tiny AdderBoard | 101 | 35 / 37 | #100: 82 / 71 | #100: 82 / 71 | 126 ms | 18–23 ms |
| UCI HAR | 201 | 14 / 12 | #200: 39 / 16 | #200: 39 / 16 | 253 ms | 30–37 ms |

All 16 selected-frame checks passed the identity assertions. Fifteen returned
nonempty graphs. KWS #57 correctly retained its incomplete attempt, without
fabricating a candidate or architecture. The six sampled runs had no normalizer
diagnostics or unresolved recorded parents. Timings are one local smoke pass,
not a browser benchmark or a latency guarantee.

Every nonempty sampled graph explicitly reported `static_module_structure`
with partial AST extraction. These nodes and edges describe supported source
declarations and explicit dependencies. They are not a complete traced tensor
graph; node counts are not model parameter counts. Repeated declaration
templates, symbolic dimensions, source helpers, and runtime control flow retain
their extraction limitations.

The measured run-revision digests were:

| Task | Replay revision |
| --- | --- |
| Addition | `34a70a56cbb9306fd4e22870d8a196de0d7b4ba4c17a4f0b09fb37b4f0f9079e` |
| Fashion-MNIST | `057560d9439b05a4b28c6378219730489df4bfbad7ba693f8e3265c3b080ffdc` |
| nanoGPT | `f7f70e2543106ecee56b39ed98423d6d4a94adb00db6856f0a7b257d09d39bb2` |
| KWS | `13df1f2c9eb46af3fd3a7ec12796d60a8e781ee9161876c1c1ac56d266804fa8` |
| Tiny AdderBoard | `ddd8733479186d446294fb05e81e119d93c5797dc65157a2971c46271298729b` |
| UCI HAR | `b4273ae83e507014c7ffa86959ee84f46a33a210a81115ff6cb4da83c5c42f3a` |

## Missing and unsupported evidence

An independent full normalization of all 68 directly tracked comparable runs
found 7,615 seed/opportunity stops: 7,521 completed proposals, 26 incomplete
opportunities, and 68 seeds. Saved source resolved for 1,889/1,895 Addition,
4,020/4,020 Fashion, 671/671 nanoGPT, and 1,006/1,029 KWS occurrences. There
were no normalizer diagnostics. Source presence does not establish that every
program parses or yields a supported architecture graph.

Both known Tiny exceptions were checked through the actual API facade:
B02-C2 proposal 92 and B05-C2 proposal 98. Their saved source is available,
but Python parsing fails with `SyntaxError`; both return zero graph nodes,
`extraction.completeness: unavailable`, and a parse warning. Their categorical
annotations remain `assumed`, with no fingerprint. Carried-forward categorical
totals are not presented as proof that topology stayed unchanged.

Interrupted recovery requires stronger identity evidence than an artifact
reference. An `infrastructure_interruption` alias to a parent can resolve only
when every declared editable file in that opportunity's saved proposal
workspace is byte-identical to the parent. Missing files or changed bytes
produce `unresolved_recovery_fallback`, no source hash, and an empty graph.
The raw invalid-launch KWS history has 34 such unresolved completed attempts;
their fallback parent programs are not shown as attempted architectures.
The full HAR audit found 4,019 available sources among 4,020 occurrences, with
one unresolved recovery fallback. The separate source inventory documents its
archive verification and complete normalized counts.

## Automated contracts and source consistency

The targeted backend suite passed **27 tests**:

```sh
python -m pytest -q \
  "architecture trajectory visualization/tests/test_architecture_replay.py" \
  "architecture trajectory visualization/tests/test_architecture_api.py"
python -m ruff check \
  "architecture trajectory visualization/architecture_trajectory_visualization/replay.py" \
  "architecture trajectory visualization/architecture_trajectory_visualization/api.py" \
  "architecture trajectory visualization/tests/test_architecture_replay.py" \
  "architecture trajectory visualization/tests/test_architecture_api.py"
```

Coverage includes repeated candidate occurrences, seed/incomplete preservation,
result-file fallback, exact large integer metrics, unresolved parents, path and
archive traversal rejection, LFS pointer detection, extraction checksums,
scope isolation, prepared receipt requirements, raw revision preservation,
source-cache envelope identity, portable export validation, read-only HTTP
routes, and revision rejection after source edits.

Missing, differing, and matching interrupted workspaces are tested separately.
Run loading, HTTP snapshots, and portable CLI/API exports use the same identity
gate. Changing a saved editable workspace invalidates the run cache and makes
an old pinned revision fail. Missing-source export frames carry null source
hashes and null source IDs, matching browser import validation.

The static frontend review checked that recorded-parent comparisons are
separate from visual transitions, ancestry uses occurrences rather than
deduplicated candidate IDs, replay routes preserve recorded parent order,
retry metrics remain separate, and categorical cumulative values retain their
published full-run basis. Catalog request sequencing and omitted semantic
operation/repeat/output fields were identified and corrected in the frontend.
The empty-source badge now distinguishes unsupported extraction from absent
source. Browser verification should cover rapid scope/run changes in addition to normal
scrubbing and playback; the pure module tests do not simulate network races.

## Sampled extraction across every prepared comparable run

A second audit covered all **108 runs**, using the same six active data
revisions described above, before the filtered archive was installed. For each
run it selected the seed, the final recorded incumbent occurrence, the last
completed occurrence, and a completed source nearest the chronological
midpoint. The midpoint preferred a source digest different from the seed;
different source bytes do not by themselves establish a different architecture.
Final incumbents were resolved by recorded incumbent changes, preserving the
occurrence at which that incumbent was established.

The 432 role assignments reduced to **393 distinct selected occurrences**.
Snapshots were requested through `ArchitectureStore.snapshot` with each run's
revision pinned, and extraction was deduplicated by source digest across runs.
This required **289 unique descriptor snapshots**, including one empty-source
descriptor reused for missing sources. The audit finished in 23.2 seconds on
this machine; this is a local measurement, not a performance guarantee.

| Task | Runs | Selected occurrences | Available source / nonempty graphs | Unsupported saved source | Missing or unresolved source | Nodes range | Edges range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Addition | 16 | 61 | 61 / 61 | 0 | 0 | 42–141 | 35–121 |
| Fashion-MNIST | 20 | 77 | 77 / 77 | 0 | 0 | 13–98 | 14–121 |
| nanoGPT | 12 | 47 | 47 / 47 | 0 | 0 | 34–43 | 17–24 |
| KWS | 20 | 71 | 68 / 68 | 0 | 3 | 9–26 | 3–24 |
| Tiny AdderBoard | 20 | 74 | 74 / 74 | 0 | 0 | 35–164 | 34–207 |
| UCI HAR | 20 | 63 | 63 / 63 | 0 | 0 | 9–192 | 0–243 |

Ranges describe nonempty descriptors and include all typed edges, including
containment and parameter ties. Three HAR samples had supported declarations
but zero supported connections; the extractor left their topology unspecified.
No nonempty sampled descriptor had an unresolved model-construction entry
point. That does not make its helper calls or runtime topology resolved.

All unique descriptors passed these assertions:

- Node, edge, and group IDs were unique within each descriptor; every edge
  endpoint and node group reference existed.
- Every emitted node and edge carried source evidence, and edge types remained
  explicit rather than turning containment or parameter ties into tensor flow.
- The snapshot envelope matched the requested occurrence, candidate, source
  digest, source revision, and pinned run revision.
- Python extraction used `static_module_structure`, with `partial`
  completeness for nonempty graphs and `unavailable` for empty graphs.
  Its note identified class groups as definitions, not expanded instances.
- Descriptor parameter totals stayed null rather than claiming to recompute
  unique weights, and all descriptors serialized as strict finite JSON.

There were **zero assertion failures or extraction exceptions**. Among the 393
selected occurrences, 373 carried one or more extraction warnings. These
included unresolved calls in 299 samples, unexpanded control flow in 226,
declaration templates in 67, repeated-module invocation limits in 29, and
unresolved parameter-alias owners in 11. Warnings can overlap. These contract
checks establish consistent, conservative reporting; they do not prove that
the partial schematic captures every computation in the candidate program.

The three selected missing sources were KWS B01-C1 proposal 57, B03-C1 proposal
47, and B04-C0 proposal 59. Each yielded no graph instead of substituting a
parent's program. The B03-C1 record was also the final recorded incumbent.
The two known Tiny syntax errors and HAR's interrupted recovery exception
described earlier were outside this four-role sample, so the zero unsupported
count in this table does not negate those known exceptions.

Run normalization performed for the audit also counted **13,625 available
sources among 13,655 total occurrences** across the 108 runs: Addition
1,889/1,895; Fashion 4,020/4,020; nanoGPT 671/671; KWS 1,006/1,029; Tiny
2,020/2,020; HAR 4,019/4,020. Only the selected occurrences received the graph
checks above. Full occurrence, role, source digest, warning, and run-revision
records were saved locally in
`outputs/architecture-viewer/extraction-audit-108-runs.json`.
That historical artifact was subsequently reproduced with the filtered receipt
excluded, retaining the same selected occurrences and counts; its recorded
25.6-second timing is the reproduction, while 23.2 seconds above is the
original measurement.

## Final prepared catalog and regression checks

After the measurements above, preparation of the filtered archive completed.
All three archive payloads passed their full SHA-256 checks. The API now selects
verified prepared roots for all six campaigns; older directly tracked histories
remain separately accessible in archive scope. The earlier measurements remain
identified with their original revisions and are not combined with these newer
histories.

The final independent normalizer audit recorded:

| Task | Runs | Seed + proposal occurrences | Terminal proposals | Available saved source | Final proposal per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Addition | 16 | 1,936 | 1,920 | 1,936 | 120 |
| Fashion-MNIST | 20 | 4,020 | 4,000 | 4,020 | 200 |
| nanoGPT | 12 | 732 | 720 | 732 | 60 |
| KWS | 20 | 4,020 | 4,000 | 4,020 | 200 |
| Tiny AdderBoard | 20 | 2,020 | 2,000 | 2,020 | 100 |
| UCI HAR | 20 | 4,020 | 4,000 | 4,019 | 200 |
| **Total** | **108** | **16,748** | **16,640** | **16,747** | — |

There are 108 seeds and no incomplete opportunities in these prepared
comparable histories. The only unavailable saved source is HAR B02-C3 proposal
171, an unresolved interrupted recovery fallback. Available saved source does
not imply valid Python: the two documented Tiny syntax exceptions remain.
Full status counts, source availability, and missing-source identities are in
`outputs/architecture-replay-data/normalized-inventory.json`.

A fresh first/last snapshot check then used B01-C0 from all six final prepared
campaigns. All 12 requests passed occurrence/source/revision checks and yielded
nonempty partial schematics. The newer nanoGPT endpoint is proposal 60
(34 nodes, 17 edges), and the newer KWS endpoint is proposal 200 (11 nodes,
6 edges). Full selected-frame digests and dimensions are saved in
`outputs/architecture-replay-data/final-endpoint-verification.json`.

The complete new architecture and preparation Python suite passed **58 tests
and 2 subtests** in the final check:

```sh
python -m pytest -o addopts='' -q "architecture trajectory visualization/tests"
```

The focused legacy dashboard Python suite passed **45 tests**, with one
pre-existing literal assertion failure:
`test_page_contains_live_controls_and_raw_outcome_overlay` expects
`seriesMode:'runs'` verbatim. The original source already uses
`seriesMode:payload?.combined_all?'conditionMean':'runs'`. Executing the original
test function against the original HTML from Git commit
`65f4aac20963bac4d293879c4e3c2ce96070edfb` reproduced the same failure.

Eight of the nine legacy `tests/dashboard_*.cjs` scripts passed. The existing
`dashboard_initial_window.cjs` script fails in its isolated VM context with
`ReferenceError: isOntologyMetric is not defined`. Running the original script
against the original HEAD HTML in a temporary directory reproduced that same
failure. Neither legacy failure was introduced by architecture replay, and
neither was hidden by changing its assertion or test harness. These results do
not claim that the entire repository test suite is clean.

## Final sampled extraction across all 108 prepared runs

The four-role extraction audit was repeated after all archives were prepared,
using the seed, final recorded incumbent, last completed occurrence, and
completed source nearest the midpoint of every comparable run. Source choices
and incumbent resolution followed the method in the earlier sampled audit.
All snapshots used `ArchitectureStore` with the corresponding run revision
pinned; no candidate program was imported or executed.

The 432 role assignments reduced to **397 selected occurrences** and **295
distinct source descriptors** after deduplicating source digests across calls.
All 397 selected occurrences had available source and returned nonempty
`python_ast` descriptors marked `static_module_structure` and `partial`.
There were **zero extraction exceptions or integrity assertion failures**.

| Task | Runs | Selected occurrences | Distinct source descriptors | Nodes range | Edges range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Addition | 16 | 59 | 44 | 42–141 | 35–121 |
| Fashion-MNIST | 20 | 77 | 58 | 13–98 | 14–121 |
| nanoGPT | 12 | 48 | 37 | 34–49 | 17–30 |
| KWS | 20 | 76 | 57 | 6–45 | 1–41 |
| Tiny AdderBoard | 20 | 74 | 55 | 35–164 | 34–207 |
| UCI HAR | 20 | 63 | 44 | 9–192 | 0–243 |
| **Total** | **108** | **397** | **295** | — | — |

Every unique descriptor passed the earlier ID, edge-endpoint, group-reference,
source-evidence, occurrence/candidate/revision, finite-JSON, and conservative
completeness assertions. Parameter totals remained null in descriptors, leaving
recorded measurement totals separate. No sampled descriptor had an unresolved
model-construction entry point. There were zero missing-source or unsupported
saved-source results **within this sample**.

The selected graphs remain partial: **377 of 397** carried warnings. These
included unresolved calls in 319 samples, unexpanded control flow in 258,
unpacked outputs in 304, declaration templates in 66, repeated-module call
limits in 32, and unresolved parameter-alias owners in 11. These categories
overlap. The 76 KWS samples also described the limits of the recorded
`recurrent_step` interface. Three HAR samples had module declarations but no
supported edges; the extractor did not invent connections for them. Edge
ranges include typed containment and parameter-sharing relationships as well
as supported data flow.

The two known Tiny syntax errors and HAR B02-C3 proposal 171 were outside this
four-role selection and remain documented exceptions. This audit validates
representative descriptors from every run, **not extraction of all 16,748
occurrences** or exact reconstruction of arbitrary computation. The full
normalization performed alongside it independently reproduced the final
**16,747/16,748** saved-source count.

The final audit took 51.1 seconds locally, without a latency guarantee. Complete
sample roles, source hashes, run revisions, warnings, data origins, and counts
are saved in
`outputs/architecture-viewer/extraction-audit-108-runs-final-prepared.json`.
All source receipts identify GitHub revision
`65f4aac20963bac4d293879c4e3c2ce96070edfb`; the extractor file SHA-256 was
`8109f6fba2b2702dd0647ee69c45c049f5402ed72b68985bb5d2633d74cc4590`.

## Final browser and presentation verification

The shipped `architecture trajectory visualization/frontend/tests/browser-smoke.mjs` passed
all **11 interaction checks** against the completed prepared catalog, with
**zero browser errors**. It covered real default data, 40 rapid slider inputs,
matching occurrence metrics and actual-parent comparisons, rejected endpoints,
URL restoration, playback stopping at the last opportunity, all six task seeds,
rapid dashboard/archive scope changes, PNG capture, desktop labels, and the
390 × 844 narrow layout. Full results are saved in
`outputs/architecture-viewer/browser-verification.json`.

The 40-input scrub burst settled on proposal 39 and its matching graph in
**1,000.4 ms** in the final local run. This is a measured burst completion time,
not a per-frame performance claim. The detailed desktop view retained all 23
source nodes with 15 visible labels and zero overlapping visible labels. On
mobile, the same graph collapsed to three source-containment blocks: all three
were inside the viewport, there was no horizontal overflow, timeline controls
stayed above the chart, and the slider occupied y=797–817 within an 844px-high
window. Visual inspection confirmed the desktop and mobile screenshots.

Final browser review caught and fixed camera fitting against outgoing animation
coordinates and fitting before a resize observer updated the aspect ratio.
The fit operation now uses target coordinates and current viewport dimensions.
Mobile timeline controls use two reserved rows so proposal text cannot overlap
the metric plot. Automated checks cover both outcomes.

Frontend checks passed **20 Node tests**, JavaScript syntax checks, Prettier,
and the production build. Ruff passed for the new Python modules/tests and the
modified server, and `git diff --check` was clean. The separate browser notes in
[the independent browser notes](BROWSER_VERIFICATION_NOTES.md) document keyboard/URL interaction,
reduced motion, WebGL fallback, portable bundle import, and source-unavailable
states exercised earlier in the same implementation session.

A real Fashion-MNIST C0 block-1 replay was captured from seed through proposal
15 at 2× playback with validation accuracy selected. The resulting WebM is
**13.666 seconds, 1440 × 900, 1,058,026 bytes**. Chromium successfully decoded
it and sought to a middle frame without an error. Seed/end occurrence IDs,
metrics, and the matching run revision are recorded in
`outputs/architecture-viewer/architecture-replay-demo.json`. The clip retains
rejected outcomes, actual-parent comparisons, partial-source labels, and
GitHub provenance. JPEG capture timing was preserved during VP8 encoding;
it does not claim to show real-time training or measured activations.

Local presentation artifacts:

- `outputs/architecture-viewer/desktop-final.png`
- `outputs/architecture-viewer/mobile-final.png`
- `outputs/architecture-viewer/architecture-capture.png`
- `outputs/architecture-viewer/architecture-replay-demo.webm`
- `outputs/architecture-viewer/demo-start-frame.jpg`
- `outputs/architecture-viewer/demo-end-frame.png`
- `outputs/architecture-viewer/demo-playback-verified.png`

## Continuous architecture morphing follow-up

The renderer now reuses matched meshes and labels across iterations. Analytic
critically damped motion preserves the current pose and velocity when retargeted;
positions, schematic sizes, opacity, and material colors settle continuously.
Connections follow the moving endpoints each frame, while outgoing nodes and
connections are disposed after fading. Loading no longer hides the previous
scene: its explicit pending caption distinguishes it from the newly selected
metadata. Adjacent descriptor prefetch is bounded to four requests and pinned
to run revision. Still captures settle to the selected recorded snapshot.

All **24 frontend unit tests**, syntax checks, formatting, production build, and
**11 existing browser checks** passed. The four additional checks in
`architecture trajectory visualization/frontend/tests/browser-motion.mjs` passed with zero
browser errors:

- Real Fashion seed → proposal 1 retained 13 visual object identities, recorded
  72 active animation frames, and moved/scaled 12 nodes. The camera was exactly
  unchanged. Maximum moving-edge attachment error was `1.879e-6` world units.
- Seven rapid retargets settled on proposal 12, with the matching occurrence,
  unchanged camera, and zero remaining outgoing nodes.
- An artificial 900 ms fetch delay retained 17 visible objects, an explicit
  loading/previous-architecture caption, and disabled stale inspection/capture.
- Reduced motion committed proposal 89 immediately, with no active morph or
  outgoing nodes.

Results are saved in `outputs/architecture-viewer/browser-motion-verification.json`;
regression results are in `/private/tmp/rl4rl-morph-smoke/browser-verification.json`.
A separate local frame sample measured a 16.7 ms median animation interval and
zero hidden-canvas frames during the real seed → proposal 1 transition. This
single local measurement is not a frame-rate guarantee on other hardware.
The demo WebM and provenance JSON were regenerated with the smoother renderer:
10.833 seconds at 1440 × 900, again using real Fashion proposals 0–15. Browser
video decoding passed. The previous clip measurements above describe the prior
renderer version.

## Feature folder verification

The Python package, frontend, tests, launcher, and documentation now live in
`architecture trajectory visualization/`. The existing dashboard retains only
the route and navigation integration. The launcher resolves repository and
asset paths from its own location; catalog and command help were also exercised
from outside the repository.

After relocation, all **58 Python tests** (plus two subtests), **24 frontend
tests**, JavaScript syntax checks, Prettier, Ruff, and the production build
passed. The relocated browser scripts passed all **11 dashboard checks** and
**four motion checks** against the running server, with no browser errors.
Reports are under `outputs/architecture-viewer/relocation-check/`. Independent
review found no blocking import, path, documentation-link, or fresh-checkout
issue. Generated assets, dependencies, recordings, and datasets are excluded
from the source change.
