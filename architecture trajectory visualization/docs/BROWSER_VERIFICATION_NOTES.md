# Architecture explorer: independent browser checks

Checked locally on 2026-09-19 with Chromium 1228 through `agent-browser`,
session `rl4rl-arch-audit`, at `http://127.0.0.1:8765/architectures`.
The checks used real saved candidates and a CLI-exported local JSON bundle.
No candidate code was executed, and no bundle was uploaded to a server.

This report covers the independent mobile, accessibility preference, fallback,
bookmark, and import pass. The separate desktop smoke script and Python/JS
tests cover other behavior.

## Observed results

| Check | Evidence |
| --- | --- |
| Narrow viewport | At 390 × 844, document dimensions were exactly 390 × 844 with no horizontal scrolling. The timeline occupied y=690–844 and the slider y=797–817. The final Fashion-MNIST overview showed three readable source groups/input blocks with no overlap or clipping. |
| Mobile source grouping | A 23-node Fashion-MNIST descriptor was projected as the actual `features` container (14 members), `classifier` container (6 members), and image input. Raw descriptor identity and selection stayed intact. |
| Reduced motion | `prefers-reduced-motion: reduce` was active. Moving backward from proposal 192 to 189 kept graph occurrence and selection aligned; projected positions stayed fixed across a 650 ms observation. |
| Bookmark restoration | Reload restored Fashion-MNIST block 1/C0, ancestry endpoint 192, proposal 189, index 41 of 43, comparison 187, the same run revision, and the same measured metrics. |
| WebGL fallback | With `webgl=off`, no canvas was present and the component list was visible. Fit, camera, and PNG controls were disabled. The selected proposal and graph occurrence stayed aligned, with no horizontal overflow. |
| Local bundle import | A CLI-exported Addition run opened at proposal 117, index 52 of 53, with graph occurrence 117, comparison 116, 140 raw nodes, validation accuracy 0.9997, and recorded parameter count 1,585. The UI stated that nothing was uploaded. |
| Import network behavior | After clearing the request log before choosing the file, the only observed request was the existing local `GET /api/revision` poll; no upload or snapshot request was made. |
| Bundle bookmark reload | Reload displayed an explicit request to choose the matching local bundle again, with no stale architecture and a disabled slider. Reopening the file restored proposal 117, the same revision, and metrics. |
| Empty filtered run | Addition C0/block 3 in dashboard scope returned zero runs. Selection length and node count were zero, metrics were empty, the slider was disabled, the heading said “No prepared run,” and the old metrics, difference, and inspector panels were cleared. |
| Repeated run changes | Six consecutive UI selections (Fashion-MNIST block 1: C1 → C2 → C3 → C0 → C2 → C1) settled on C1 proposal 198, index 54 of 55. The selected run, graph occurrence, and metrics agreed: validation accuracy 0.9267 and parameters 233,434. |
| Malformed bundle | An invalid `architecture-bundle/1` object was rejected with “Expected architecture-bundle/1 with a versioned run and snapshots.” The prior Fashion-MNIST run, graph, and metrics stayed unchanged. The valid CLI bundle was then accepted again by the final stricter validator. |
| Missing candidate source | Keyword Spotting block 1/C0, All attempts, last proposal 57 retained its occurrence and parent comparison 56. The graph had zero nodes and zero labels, metrics were empty, and the UI explicitly said “Source unavailable” with a missing-snapshot explanation. |

## Screenshots

Screenshots are local generated artifacts under `outputs/architecture-viewer/`:

- `audit-mobile-final.png`: visually inspected final 390 × 844 grouped scene.
- `audit-mobile-fallback.png`: component-list fallback without WebGL.
- `audit-bundle-bookmark.png`: local-bundle bookmark asking for its file.

Earlier `audit-mobile-390.png` and `audit-mobile-fit.png` captured a label
overlap/clipping issue before the mobile grouping fix; they are not final
acceptance screenshots.

## Corrections found during the pass

The initial narrow scene retained desktop framing and had overlapping labels.
The final build refits at the responsive breakpoint and groups source
containers. A later empty-timeline render assumed `warnings` was present;
the implementation added an optional guard and cleared selection while loading
another run. The follow-up empty-filter, repeated-switch, missing-source, and
bundle-validation checks above passed after the final service restart. A final
390 × 844 screenshot and DOM measurement confirmed that all three displayed
label boxes remained within the viewport and did not intersect one another.

The isolated `agent-browser errors` command returned an empty error-shaped
response in this environment, so this pass does not claim a clean console from
that command. Root's separate clean-browser smoke test covers runtime errors.

These observations validate the tested UI paths, not arbitrary runtime graph
reconstruction or universal browser compatibility. Static extraction remains
partial and source backed; parameter totals remain recorded metrics.
