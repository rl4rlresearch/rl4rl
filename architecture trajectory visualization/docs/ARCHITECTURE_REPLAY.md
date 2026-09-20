# 3D architecture replay

The architecture explorer is a read-only page in the existing RL4RL dashboard:
`http://127.0.0.1:8765/architectures`. It replays saved proposal occurrences,
recorded ancestry, measured outcomes, and evidence-backed architecture
schematics. Candidate Python is parsed as text and is never imported, executed,
traced, trained, or evaluated by this feature.

The implementation starts from GitHub source revision
`65f4aac20963bac4d293879c4e3c2ce96070edfb` of
[rl4rlresearch/rl4rl](https://github.com/rl4rlresearch/rl4rl). The dataset and
known coverage differences are recorded in the
[source inventory](ARCHITECTURE_REPLAY_SOURCE_INVENTORY.md).

## Build and open

Use Python 3.11 or newer and a current Node.js/npm installation. Run these
commands from the repository root:

```sh
npm --prefix "architecture trajectory visualization/frontend" ci
npm --prefix "architecture trajectory visualization/frontend" run build
python "architecture trajectory visualization/run.py" serve --host 127.0.0.1 --port 8765
```

Open [Architecture explorer](http://127.0.0.1:8765/architectures). The existing
trajectory, ontology, scientific-process, and transcript pages remain available
on the same server. The frontend is vanilla JavaScript and Three.js, bundled by
esbuild. It has no separate application server, database, model-provider API,
or external runtime service. Build the assets again after frontend edits.

The launcher resolves the repository and feature paths from its own location.
Use another `--port` if 8765 is already occupied. Windows saved-data viewing
remains supported by the existing dashboard; its live campaign controls remain
separate from architecture replay.

## Prepare real trajectory data

Directly tracked campaign directories under `data/c0c3/` are discovered
automatically. The source revision includes Addition, Fashion-MNIST, nanoGPT,
and Tiny Keyword Spotting histories. Tiny AdderBoard and UCI HAR are supplied
in tracked Git LFS archives. The filtered archive also contains later versions
of several histories than the directly tracked directories.

Prepare all three archives from the pinned GitHub revision with the download
and extraction helper. Git LFS does not need to be installed:

```sh
python "architecture trajectory visualization/run.py" prepare --campaign all --workers 32
```

Use `--campaign tiny`, `har`, or `filtered` to prepare one archive. The helper
resumes downloads, verifies the complete archive SHA-256, safely extracts
metadata and candidate sources, and records the source receipt and provenance
marker. It writes the separate prepared roots shown below. Downloads can be
large; the source inventory records their sizes.

For an existing local archive, use the lower-level `prepare-archive` command.
You can obtain payloads with Git LFS if preferred. A small file starting with
`version https://git-lfs.github.com/spec/v1` is a pointer, not an archive;
preparation detects this case explicitly.

```sh
git lfs pull --include='data/rl4rl-c0c3-dashboard-filtered-20260906.zip,data/rl4rl-tiny-adderboard-through-100-c0-c3-20260908.zip,data/rl4rl-uci-har-c0-c3-20260908.zip'
```

`prepare-archive` preserves ZIP member paths. Inspect the member prefix before
choosing its destination: a ZIP already containing `data/c0c3/...` should be
extracted into its revision root; a ZIP starting with `<campaign>/...` should
be extracted into that revision root's `data/c0c3/` directory. For example:

```sh
python "architecture trajectory visualization/run.py" prepare-archive \
  data/<archive-filename>.zip --destination <extraction-directory>
```

Use `filtered`, `tiny`, or `har` as the revision root under
`outputs/architecture-replay-data/`, corresponding to the three archives above.
The exact archive identities and sizes are in the source inventory. The generic
CLI prints its extraction/checksum receipt; it does not relocate ZIP members or
automatically write a repository-root provenance marker.

The CLI obtains expected checksums from recorded transfer metadata or the
Git-tracked LFS pointer. An explicit `--sha256` can supply a separately verified
expected checksum. It verifies SHA-256, rejects escaping paths and symlinks,
checks ZIP members during reading, and refuses to overwrite different existing
artifacts. By default it extracts textual metadata and Python sources needed
for replay, omitting checkpoint binaries. `--all-files` includes other assets
and is unnecessary for the explorer.

Prepared data lives in separate repository-shaped roots:

```text
outputs/architecture-replay-data/
  filtered/
    architecture-replay-source.json
    data/c0c3/<campaign>/...
  tiny/
    architecture-replay-source.json
    data/c0c3/tiny-v21/...
  har/
    architecture-replay-source.json
    data/c0c3/uci-har-pareto-v21/...
```

The download helper writes `architecture-replay-source.json` at each
revision root automatically. When using the lower-level extraction command,
write this marker after verified extraction, recording
`source_revision`, `archive_filename`,
`archive_sha256`, `checksum_verified`, `prepared_at`, `data_root`, and
`archive_scope` from the actual source checkout and successful preparation
receipt. `data_root` is `data/c0c3`. Do not mark an archive verified before the
checksum check passes. Preserve this marker with the restored data: it carries
provenance for a root that is not itself a Git checkout.

The server discovers these prepared roots and prefers the complete
prepared campaign revision over its older tracked copy. It does not splice
individual runs from conflicting revisions or overwrite tracked artifacts.

The five large ontology `final.json` files are not required for compact
categorical annotation. The explorer reads the checked-in working schemas,
publication manifest, and `trajectory-metrics.json` files instead.

## Inspect a trajectory

Choose a campaign, optional framework/condition/block filters, and run. The three replay modes have
different meanings:

| Mode | Timeline contents |
| --- | --- |
| Ancestry path | Recorded ancestors of the selected endpoint. Select a parent route when a candidate has multiple parents. |
| All attempts | Every seed/proposal occurrence in recorded order, including rejected, failed, and incomplete attempts. |
| Incumbent history | Recorded incumbent changes. Portfolio retention alone does not mean a new incumbent. |

The path endpoint selector, history search/list, and ancestry overview select
recorded occurrences. A repeated source or candidate ID does not erase later
proposal occurrences. Chronological neighbors are not automatically parent and
child. A rejected branch can be inspected without assigning it to the next
candidate's lineage.

The bottom timeline remains visible while scrubbing. It includes first/last,
previous/next, play/pause, playback speed, outcome markers, and a selected
metric plot. One stop means one proposal occurrence plus the explicit seed,
not a training epoch. Scrubbing pauses playback. The selected architecture,
metadata, metrics, comparison, and plot cursor share one selection. Where
separate evaluations are recorded, the evaluation selector keeps their
metrics distinct.

Use **Recorded parent** or **Run baseline** to change the structural comparison.
On a merge, the selected parent route controls the visual parent comparison.
Published categorical primary-parent metrics retain their original comparison;
a different visual comparison does not rewrite them.

Orbit, zoom, or pan the viewport; use **Fit model** and **Top view** for a stable
framing. Select a block to inspect constructor configuration and source file/
line evidence. **Component list** provides keyboard-accessible inspection and
the fallback when WebGL cannot render. Standard focused range-input keyboard
controls are available for the timeline. Reduced-motion preferences suppress
unnecessary animation.

Iteration changes morph persistent components with continuous motion, including
when a scrub interrupts the previous transition. Added components grow into
place, removed components fade, and connections follow the moving blocks.
Normal playback uses an 850 ms spring timescale and a 1300 ms cadence after
loading, both adjusted by playback speed. Adjacent snapshots are prefetched;
a slow load keeps the previous architecture visible with an explicit pending
caption. PNG captures finish the transition to the selected recorded snapshot.

Small screens collapse supported source containers into labeled groups to keep
the overview readable. Select a group and use **Expand components** to inspect
its members; the group is source containment, not an additional tensor layer.
Add `?webgl=off` to the page URL to exercise the component-list fallback.

**Dashboard scope** retains the original comparable-run exclusions and Addition
proposal cap. **Archive history** exposes extra recorded history when present.
The active scope, source revision, and missing-data diagnostics matter when
comparing a screenshot with a published result. The URL stores replay selection
for reload/bookmark use; a local bundle bookmark is not an uploaded dataset.

## Portable bundles and API

Use **Export run** to download a self-contained JSON replay and **Open bundle**
to inspect it through the built viewer. The bundle contains normalized records
and static descriptors; it does not execute candidate code. The viewer assets
still need to be served locally, but an imported bundle does not require a
research backend or a network connection for its snapshots.

The preparation/export CLI also works on either the original checkout or one
prepared root. Use the catalog to find exact campaign and run IDs:

```sh
python "architecture trajectory visualization/run.py" catalog --scope dashboard

python "architecture trajectory visualization/run.py" \
  --repo outputs/architecture-replay-data/tiny \
  catalog --scope archive

python "architecture trajectory visualization/run.py" \
  --repo outputs/architecture-replay-data/tiny \
  export --campaign tiny-v21 --run '<exact-run-id>' \
  --scope dashboard --output outputs/tiny-replay.json
```

Replace `<exact-run-id>` with a catalog value. CLI export prepares the replay
and extracted graphs. API export additionally carries any compact categorical
annotations attached by the live server.

[`api.py`](../architecture_trajectory_visualization/api.py) supplies these
read-only endpoints:

| Endpoint | Query parameters | Result |
| --- | --- | --- |
| `/api/architectures/catalog` | `scope=dashboard` or `archive` | Available campaign/run inventory and data origins. |
| `/api/architectures/run` | `campaign`, `run`, `scope` | Normalized occurrence history and annotations. |
| `/api/architectures/snapshot` | `campaign`, `run`, `proposal`, `scope`, optional `revision` | Selected static architecture descriptor and provenance. |
| `/api/architectures/export` | `campaign`, `run`, `scope`, optional `revision` | Portable run bundle with snapshots keyed by occurrence ID. |

Use catalog-returned campaign keys in HTTP requests. Source text is resolved
only from recorded candidate locations, not arbitrary caller-supplied paths.
Descriptors are cached by source digest and extractor revision; slider movement
does not fetch candidate files from GitHub.

The viewer pins snapshot and export requests to the loaded run's data revision.
If the source data changes, a mismatched revision requires reloading the run;
older measurements cannot acquire a newer source graph silently.

## Data contracts and evidence limits

Portable bundle schema `architecture-bundle/1` has `run` and `snapshots`.
Snapshots are keyed by occurrence ID. Keep these identities separate:

- **Occurrence:** campaign, run, proposal, recorded candidate ID, sequence,
  recorded parents, outcomes/evaluations, source metadata, and lifecycle state.
- **Source:** resolved artifact/source candidate ID, file SHA-256 mapping,
  source bundle hash, and resolution diagnostics. An event ID can differ from
  its source directory's ID.
- **Descriptor (`1.0`):** `nodes`, typed `edges`, containment `groups`, source
  hash, extraction method/completeness, evidence, and warnings. A node carries
  a stable ID, operation kind, label, configuration, and source/IR evidence.
- **Difference (`1.0`):** node matches, added/removed IDs, changed fields, and
  added/removed typed connections. Duplicate structural matches can be
  ambiguous and are marked as such.

Python extraction is a useful conservative schematic, **not exact arbitrary
computation reconstruction**. Source class groups are definitions, not expanded
instances. Reachability follows resolvable model construction/reference sites;
unresolved entry points are labeled and can include unused alternatives.
Literal/default constructor values are shown where known. Unresolved
expressions, tensor shapes, runtime branches, loop counts, helper internals,
and unsupported calls remain unknown.

Only source-proven direct call dependencies and declared `nn.Sequential`
ordering become tensor-flow edges. `ModuleList` declaration order does not.
Repeated module calls and unsupported control flow are not silently collapsed
into a fabricated execution graph. KWS `recurrent_step` is a partial external
interface; its evaluator-controlled recurrent loop is not reconstructed.
Serialized `architecture_tensor_graph` IR can preserve recorded shapes and
typed graph edges, but is not re-executed to verify them.

Containment edges and shared-parameter ties are different from data flow.
Weight sharing does not establish recurrent execution. Displayed parameter
counts come from recorded metrics with their original units; the extractor
does not recompute a total or claim complete unique-weight accounting. Node
geometry uses schematic scaling, not literal parameter volume. Block width uses
a bounded logarithmic encoding when an explicit numeric output dimension is
available, and neutral sizing for unresolved expressions. Labels that
would overlap are hidden until there is room or their block is hovered or
selected. Every component remains available in the component list and
inspector; zooming in reveals more labels.

Compact categorical annotations have `reviewed`, `assumed`, `unreviewed`, or
`out_of_scope` status. Publication files are checked against manifest SHA-256
and schema versions. Joins require campaign/run/proposal/candidate identity.
Current compact publications have no source-file digest tying their review
directly to the selected source bundle, so joined annotations explicitly say
source verification is unavailable. A verified publication checksum is not a
new source audit.

Categorical fingerprints and family IDs describe reviewed mechanism categories;
they do not supply missing tensor edges or dimensions. Cumulative novelty and
retention values retain the published full-run history and are not recalculated
when a viewer selects another path. `no_change_assumed` rows carry published
metric totals but have no invented fingerprint, family, or architectural
equivalence claim. HAR remains unreviewed where the publication explicitly
excludes it.

## Capture a demo

1. Select a real run with useful source coverage and an interesting measured
   change. Choose the replay mode and endpoint before recording.
2. Use **Fit model** or **Top view**, select an informative metric, and use
   **Present** to hide secondary panels while retaining essential labels.
3. Choose a playback speed and return to the first desired stop. Record the
   browser window with the operating system's screen recorder, then press Play.
4. Use **Capture PNG** for a still image. For a recording that includes the
   entire dashboard and controls, record the browser rather than only the WebGL
   canvas.

Keep partial-structure, source, metric-unit, and unknown-result labels visible
in edited clips. Animated intermediate frames are visual transitions between
recorded snapshots, not additional evaluated models. There are no measured
activation traces and no automatic video encoder in the core replay workflow.

An optional recording helper is also included. With `agent-browser` and an
FFmpeg build supporting MJPEG input and VP8/WebM output available, run:

```sh
node "architecture trajectory visualization/frontend/tests/record-demo.mjs"
```

Set `AGENT_BROWSER`, `FFMPEG`, or `BROWSER_EXECUTABLE` to executable paths when
they are not on your normal PATH. The helper records the real Fashion-MNIST
B01-C0 run from its seed through proposal 15 at 2× playback, using validation
accuracy. It writes a WebM and a JSON provenance record under
`outputs/architecture-viewer/` and verifies browser decoding. JPEG capture
timing is preserved in the encoded clip; this is a screen recording of replay,
not a recording of training or measured activations.

## Development checks

Run the focused Python and frontend checks from the repository root:

```sh
python -m pytest "architecture trajectory visualization/tests" -q
npm --prefix "architecture trajectory visualization/frontend" test
npm --prefix "architecture trajectory visualization/frontend" run check
npm --prefix "architecture trajectory visualization/frontend" run format:check
npm --prefix "architecture trajectory visualization/frontend" run build
```

With the dashboard running and `agent-browser` available, run the interaction
checks from the viewer directory:

```sh
cd "architecture trajectory visualization/frontend"
npm run test:browser
npm run test:motion
```

`AGENT_BROWSER` and `BROWSER_EXECUTABLE` can select local executables. Results
and desktop/narrow screenshots are written to `outputs/architecture-viewer/`.

`npm run check` performs JavaScript syntax checking; this frontend does not
claim TypeScript type checking. Browser verification should exercise actual
campaigns, a long run, parent-route changes, rapid scrubbing, incomplete source,
bundle round-trips, narrow layouts, capture, and reduced-motion/WebGL fallback.
Record those outcomes separately; a successful build does not prove them.
