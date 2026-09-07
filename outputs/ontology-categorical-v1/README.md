# Published categorical ontology results

These publications contain the reviewed categorical component fingerprints, exact
architectural family IDs, proposal lineage, metric counts, and source-review
evidence for five campaigns. UCI HAR is intentionally excluded while its review
is unfinished. Intermediate worker files and caches are not part of this publication.

## Download after pulling

Install Git LFS on your machine, then run these commands from an existing clone:

```sh
git lfs install
git pull --ff-only
git lfs pull --include="outputs/ontology-categorical-v1/forks/*/final.json" --exclude=""
```

With Git LFS installed, a normal clone or pull retrieves these JSON files
automatically. If LFS smudging was disabled, the explicit `git lfs pull` command
downloads them. A small file beginning with `version https://git-lfs.github.com/`
is an LFS pointer, not the research data.

## Included records

| Campaign | Runs | Classified candidate IDs | Reviewed proposal/seed records | Unclassified records |
| --- | ---: | ---: | ---: | ---: |
| [Addition](forks/addition/final.json) | 16 | 1,875 | 1,936 | 0 |
| [Fashion-MNIST](forks/fashion/final.json) | 20 | 3,705 | 4,020 | 0 |
| [Speech / KWS](forks/kws/final.json) | 20 | 3,940 | 4,020 | 0 |
| [nanoGPT](forks/nanogpt/final.json) | 12 | 702 | 732 | 0 |
| [Tiny AdderBoard](forks/tiny_adderboard/final.json) | 20 | 1,985 | 2,018 | 2 |

Candidate identities are classified once; repeated proposal occurrences remain
separate trajectory records. Tiny AdderBoard retains two syntax-invalid source
exceptions in `unresolved_occurrences`: B02-C2 proposal 92 and B05-C2 proposal 98.
They have no reviewed fingerprint or family. Its compact dashboard view explicitly
tags zero-change assumptions as `no_change_assumed`; these assumptions must not
be interpreted as successful classifications.

## File contents

Each campaign directory contains:

- `final.json` (Git LFS): the full publication. `runs[run_id]` contains proposal
  records with `fingerprint`, `family_id`, `family_components`, `changed_components`,
  `ontology_parent_id`, `retained`, and the eight base metrics. The publication
  also embeds its schema, source reviews, review audits, and recovery provenance.
- `working-schema.json`: the campaign schema matching the publication revision.
- `trajectory-metrics.json`: a derived dashboard view with implemented/retained
  metrics and categorical vectors, without the large review-evidence bodies.

The [publication manifest](publication-manifest.json) records exact coverage,
schema revisions, file sizes, and SHA-256 checksums. JSON bytes are preserved
across operating systems for checksum verification. Local paths inside review
provenance identify the original review environment; the embedded fingerprints
and evidence can be read without those paths existing on your machine.

## Use in the dashboard

These files are already at the paths used by the dashboard. The original campaign
event logs under `data/c0c3` are also needed to render runs and resolve the
pre-proposal portfolio comparison. The separately tracked
[filtered campaign archive](../../data/rl4rl-c0c3-dashboard-filtered-20260906.zip)
contains the campaign data snapshot.

A fresh checkout changes file timestamps. The dashboard detects this and rebuilds
its compact view in memory from `final.json`, so the saved source timestamp is not
a portability requirement. Do not substitute an LFS pointer for a downloaded
publication. The metric definitions, retention policy, comparison choices, and
missing-data assumptions are described in the
[measurement contract](../../docs/ONTOLOGY_CATEGORICAL_FINGERPRINT_V1.md).

To inspect a publication directly:

```python
import json
from pathlib import Path

path = Path("outputs/ontology-categorical-v1/forks/nanogpt/final.json")
publication = json.loads(path.read_text(encoding="utf-8"))
record = next(iter(publication["runs"].values()))[1]
print(record["fingerprint"])
print(record["family_id"])
```
