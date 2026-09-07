# Source recovery results

All 355 previously flagged records in these four campaigns now resolve to verified stored source snapshots. All eight metrics have been rebuilt, with original proposal numbers and primary parents preserved.

| Campaign | Recovered records | Validated candidate identities | Trajectory occurrences | Runs | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Addition](../outputs/ontology-categorical-v1/forks/addition/HANDOFF.md) | 14 | 1,875 / 1,875 (100%) | 1,936 | 16 | 0 |
| [nanoGPT](../outputs/ontology-categorical-v1/forks/nanogpt/HANDOFF.md) | 10 | 702 / 702 (100%) | 732 | 12 | 0 |
| [Fashion](../outputs/ontology-categorical-v1/forks/fashion/HANDOFF.md) | 268 | 3,705 / 3,705 (100%) | 4,020 | 20 | 0 |
| [Speech / KWS](../outputs/ontology-categorical-v1/forks/kws/HANDOFF.md) | 63 | 3,940 / 3,940 (100%) | 4,020 | 20 | 0 |

Addition includes the 16 dashboard-configured C0-C3 runs through proposal 120. Occurrences include each run's seed; shared source reviews are reused across occurrences.

Recovery separates the recorded event ID from the actual source snapshot ID. Every reused review matched the complete stored Python source bundle and passed current-schema certificate validation. Speech's 25 interrupted opportunities additionally have byte-identical saved editable-workspace evidence.

The recovered Fashion rows contribute 79 component edits and 46 family switches; speech adds 2 component edits and 2 family switches. None of these recovered rows introduces newly explored component states or families. Reused snapshots are compared with their actual primary parent; failures are not uniformly assigned zero changes.

Verification: 58 regression tests passed. The final audit checks exact inventory/packet/published occurrence coverage, scope, preserved lineage, all eight marginal/cumulative metrics, current schema revisions, and removal of resolved exception cards from active queues. Historical exception cards and reports remain archived in each campaign's `source-recovery/before/` directory.

See [source resolution rules and reproduction commands](ONTOLOGY_SOURCE_RESOLUTION.md) and the [machine-readable cross-campaign audit](../outputs/ontology-categorical-v1/source-recovery-audit.json).


## Tiny AdderBoard follow-up

Recovered and revalidated nine additional missing-source cases using exact
stored snapshots. Tiny AdderBoard now has **1,985/1,987 validated candidates
(99.8993%)** and **2,018/2,020 published occurrences** across 20 C0-C3 runs,
through proposal 100.

The remaining two are syntax-invalid original submissions: B02-C2 proposal 92
and B05-C2 proposal 98. Their original preflight failures and malformed source
expressions are recorded in the [Tiny recovery audit](../outputs/ontology-categorical-v1/forks/tiny_adderboard/source-recovery/completion.json).
No hypothetical code correction or parent fingerprint was substituted.
The handoff's incorrect proposal numbers were corrected against the event logs.

Resolved exception cards, including stale cards from the original review,
were archived. The dashboard leaves the two invalid candidates unavailable
and suppresses totals/novelty for the ten later records whose histories cross
those gaps. All 53 categorical dashboard/fingerprint/review tests passed.


### Updated missing-candidate metric policy

By explicit user choice, unresolved in-scope candidates contribute zero marginal
changes and zero novelty. Dashboard totals carry forward through these records
and continue accumulating later increments. The previous-proposal comparison
carries the last state through an unresolved proposal. These metric-only records
are tagged `ontology_no_change_assumed`; they have no reviewed fingerprint or
family and do not increase classification completion. This supersedes the
previous downstream masking policy for explicitly unresolved candidates.
Out-of-scope data and missing comparison evidence remain unavailable.
