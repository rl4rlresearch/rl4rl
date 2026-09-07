# Source identity and recovery

The recorded candidate ID identifies a proposal record; it is not always the
name of a new source directory. The runner can assign a synthetic ID to a
duplicate or unsuccessful attempt while its `artifact_path` identifies the
actual stored source snapshot. Never infer unavailable source solely from a
missing `candidates/<recorded_id>` directory.

The categorical inventory resolves a recorded artifact only inside that run's
candidate store. It preserves the original candidate ID, proposal number and
all parent IDs. `source_resolution` records the separate source candidate ID,
artifact reference and evidence. Original campaign files are never modified
or copied into invented candidate directories.

A recovered review is reusable only when its complete Python source bundle
matches the recorded snapshot, its schema revision is current, and its source
certificate passes `validate_source_review`. API names, proposal descriptions,
failure labels and similar-looking code are insufficient evidence.

## Interrupted opportunities

An interruption's parent artifact can be a recovery fallback. It is not proof
that the attempted code was unchanged. Such an event remains unresolved unless
additional evidence identifies its source. The implemented recovery rule checks
every declared editable file in that opportunity's preserved
`proposal-workspace` against the recorded primary-parent snapshot. Only an
exact byte match for every file permits unchanged-source resolution. The
manifest retains the workspace path and each file's SHA-256 digest. Missing or
different workspace files do not pass this rule.

## Consequences for the eight metrics

- An unchanged parent snapshot gives zero marginal component changes, new
  component states, family changes and new families. All four cumulative totals
  remain unchanged at that proposal.
- A reused non-parent snapshot is compared with the actual primary parent.
  It can count as component and family changes. Novelty still depends on all
  earlier fingerprints in the same run, including unsuccessful proposals with
  a source-backed fingerprint.
- Proposal numbers are preserved, including failures. Source review reuse
  never removes a trajectory occurrence or redirects its primary parent.
- Truly unidentified source remains unresolved. Never substitute a parent
  merely to fill a fingerprint or force a complete metric series.

## Scope and reproduction

Addition uses the dashboard's configured run exclusions, C0-C3 only, and a
maximum proposal of 120. Its active scope is 16 runs and 1,936 seed/proposal
occurrences, representing 1,875 recorded candidate identities. Excluded runs
are B03-C0, B03-C3, B04-C2 and B05-C1. Browser checkbox choices do not define
this scope.

For an applicable campaign, run from the repository root:

```powershell
& outputs\tiny-seed-search\venv\Scripts\python.exe -m experiments.recover_ontology_artifact_sources --campaign kws
& outputs\tiny-seed-search\venv\Scripts\python.exe -m experiments.rebuild_ontology_categorical_campaign --campaign kws
```

Replace `kws` with `addition`, `fashion` or `nanogpt` as appropriate. The first
command reuses an existing source-identical review or reports a remaining
review need. The second requires complete current-schema coverage, validates
certificates, recomputes all eight metrics, and publishes the final output.
It refreshes the inventory and reconciles existing packets without changing
their IDs or deleting prior results. It preserves historical progress and
unresolved reports under each campaign's `source-recovery/before/` directory.

Use these commands for corrected campaigns instead of recreating packet
ownership or using the earlier publication scripts that skipped missing rows.
The resulting `source-recovery/completion.json` is the source-resolution audit;
`final.json` includes every scoped occurrence and its derived metrics.


## Tiny AdderBoard source failures

Tiny AdderBoard uses the same exact stored-source alias recovery rule. Its
recovery command is `python -m experiments.recover_ontology_artifact_sources
--campaign tiny_adderboard`. Nine records resolve this way. Two other records
are actual syntax-invalid submissions: B02-C2 proposal 92 (`source source.abs()`)
and B05-C2 proposal 98 (`basis @ @ @ torch.linalg.inv(...)`). They failed the
original source preflight. A hypothetical repaired source is not an exact
recovered submission and is not substituted for a source-backed classification.

Tiny's partial publisher remains `outputs/ontology-categorical-v1/forks/
tiny_adderboard/build_final.py`, followed by `source-recovery/finalize.py` in the
same campaign output directory. These refresh the inventory, validate all
available reviews, preserve the two invalid-source exceptions, reconcile packets
and write the recovery audit. See the campaign handoff for complete commands.

A missing fingerprint must not be silently skipped in an exact cumulative
count. The compact dashboard view therefore suppresses all cumulative counts
and component/family novelty values after a gap in published proposals. Known
parent-relative marginal component and family changes remain available. A
selected comparison also propagates missing increments into subsequent totals.
The complete four-campaign results are unaffected by this rule.


### Updated missing-candidate metric policy

By explicit user choice, unresolved in-scope candidates contribute zero marginal
changes and zero novelty. Dashboard totals carry forward through these records
and continue accumulating later increments. The previous-proposal comparison
carries the last state through an unresolved proposal. These metric-only records
are tagged `ontology_no_change_assumed`; they have no reviewed fingerprint or
family and do not increase classification completion. This supersedes the
previous downstream masking policy for explicitly unresolved candidates.
Out-of-scope data and missing comparison evidence remain unavailable.
