# Architecture trajectory visualization

Explore how recorded RL4RL architectures change over time. Select a run and
trajectory, scrub the bottom timeline, or play continuously morphing 3D source
schematics alongside measured outcomes and parent comparisons. Ancestry paths,
all attempts, and incumbent history preserve their distinct recorded meanings.

The feature includes the Python data API and source extractor, Three.js viewer,
tests, and documentation in this folder. It integrates with the existing RL4RL
dashboard and reads the repository's campaign data and ontology publications.
Candidate Python is parsed as text and never executed. The 3D diagrams are
explicitly partial source reconstructions; unknown topology stays unknown.

## Run

Use Python 3.11+ and Node.js/npm. From the repository root:

```sh
npm --prefix "architecture trajectory visualization/frontend" ci
npm --prefix "architecture trajectory visualization/frontend" run build
python "architecture trajectory visualization/run.py" serve --host 127.0.0.1 --port 8765
```

Open [the architecture explorer](http://127.0.0.1:8765/architectures).
The server also retains the existing trajectory and ontology pages. Rebuild
frontend assets after editing the viewer.

## Prepare the complete GitHub data

Directly tracked histories are available immediately. Prepare the three
verified GitHub archives to obtain the complete six-task comparable catalog:

```sh
python "architecture trajectory visualization/run.py" prepare --campaign all --workers 32
```

The helper resumes downloads, verifies SHA-256, and preserves each archive's
revision and receipt under the repository's `outputs/architecture-replay-data/`.
It does not require Git LFS or overwrite a conflicting research-data revision.
The pinned archives contain 108 comparable runs and 16,748 seed/proposal
occurrences. See the [source inventory](docs/ARCHITECTURE_REPLAY_SOURCE_INVENTORY.md)
for the archive sizes, coverage, exceptions, and provenance.

The same launcher exposes `catalog`, `export`, and `prepare-archive`
commands. For example:

```sh
python "architecture trajectory visualization/run.py" catalog --scope dashboard
python "architecture trajectory visualization/run.py" export --help
```

## Test

From the repository root, with the project's development dependencies installed:

```sh
python -m pytest "architecture trajectory visualization/tests" -q
npm --prefix "architecture trajectory visualization/frontend" test
npm --prefix "architecture trajectory visualization/frontend" run check
npm --prefix "architecture trajectory visualization/frontend" run format:check
npm --prefix "architecture trajectory visualization/frontend" run build
```

With the dashboard running and `agent-browser` available:

```sh
npm --prefix "architecture trajectory visualization/frontend" run test:browser
npm --prefix "architecture trajectory visualization/frontend" run test:motion
```

Set `AGENT_BROWSER` and `BROWSER_EXECUTABLE` if their executable paths are not
available automatically. Generated screenshots, recordings, and browser reports
remain under the repository's `outputs/architecture-viewer/`.

- [Usage, API, bundles, extraction limits, and demo capture](docs/ARCHITECTURE_REPLAY.md)
- [Verification results and known legacy test failures](docs/ARCHITECTURE_REPLAY_VERIFICATION.md)
- [Independent mobile, fallback, and import checks](docs/BROWSER_VERIFICATION_NOTES.md)
