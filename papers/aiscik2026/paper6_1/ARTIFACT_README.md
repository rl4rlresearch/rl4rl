# Paper 6.1 reproducibility notes

This directory contains the deterministic analysis and manuscript builder for:

**Question the Premise, Pay the Price: Scheduled Assumption Challenges in
Autonomous ML Research**

The supplemental artifact is released under the MIT License; see `LICENSE`.

## Inputs

The analysis reads already-recorded greedy OpenEvolve 2.1 campaigns:

- `data/c0c3/controlled-openevolve-transformer-v2-1-mps-campaign`
  (20 trajectories, proposals 1--80);
- `data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign`
  (20 trajectories, proposals 1--200);
- `data/c0c3/nanogpt-openevolve-v2-1-h100-campaign`
  (12 trajectories, proposals 1--40).

These task-specific horizons contain 6,080 proposals. They are analytic
horizons, not a single common truncation. The script makes no model calls,
starts no controllers, and runs no candidate evaluations.

## Verify the frozen analytic snapshot

From the repository root:

```bash
python papers/aiscik2026/paper6_1/verify_snapshot.py
python papers/aiscik2026/paper6_1/robustness.py
```

`verify_snapshot.py` validates the complete run roster, task-specific horizons,
condition mapping, intervention schedule, primary matched point estimates,
cycle decomposition, and denominators directly from the included 6,080-row
snapshot. `robustness.py` regenerates block-sign, leave-one-block-out, and
retention-conditioned descendant checks. Both scripts are independent of the
mutable live campaign directories.

`analysis.py` is the upstream raw-record reconstruction script used to create
the frozen snapshot. It validates prompt placement, reconstructs source from
parent/candidate artifacts, and uses fixed bootstrap seed `20260903` for 10,000
block-cluster resamples. It requires the corresponding raw campaign archive;
the repository's live campaign directories may later contain additional
conditions or incomplete replacement runs and are not the archival input.
Numerical CSV and JSON files are deterministic; rendered PNG metadata may vary
by environment.

Snapshot verification requires Python 3.11+. Raw reconstruction and figures
add Matplotlib. The PDF builder additionally requires ReportLab; pypdf is
useful for PDF verification.

## Generated files

- `proposal_records.csv`: one row per in-scope proposal.
- `checkpoint_pairs.csv` and `checkpoint_effects.csv`: 304 paired scheduled
  checkpoints and task/memory summaries.
- `preintervention_placebos.csv`: the same local estimator at proposals 2--9.
- `cycle_gain_pairs.csv`: immediate, follow-up, and ten-proposal gain.
- `population_dispersion.csv`: between-run lexical and family distance.
- `checkpoint_message_corpus.md`: all saved public mechanism, hypothesis,
  intended-edit, evidence, and outcome fields at 608 matched checkpoints.
- `qualitative_sample.md`: deterministic high-information trace sample.
- `overview.json` and `integrity.json`: headline results and denominators.
- `block_checkpoint_effects.csv`: one checkpoint-effect estimate per block.
- `leave_one_block_out.csv`: task-wise leave-one-block-out estimates.
- `descendant_branch_summary.csv`: descriptive follow-up gains split by
  whether the challenged proposal was retained.
- `lineage_descendant_cycles.csv` and `lineage_descendant_summary.csv`: exact
  parent-chain tracing and descendant-only gain for every checkpoint window.
- `population_measure_sensitivity.csv` and
  `population_measure_sensitivity_checkpoints.csv`: full-rationale,
  mechanism-only, primary-family, and lexical population-dispersion views.
- `source_missingness.csv`: arm- and task-specific source denominators.
- `robustness_summary.json`: exact sign-test and leave-one-block-out summaries.
- three rendered figures.

`qualitative_audit.md` records the human-readable trace analysis used in the
paper. `section_expansion.md` is an author-facing evidence bank with additional
results and examples for every manuscript section. `related_work_audit.md`
records the contribution boundary against the closest work, and
`claim_evidence_map.md` maps headline statements to frozen evidence files.

## Claim boundary

The strongest estimates are local matched difference-in-differences around
scheduled challenge checkpoints. Conditions share a block seed but were not
randomized and are not exact shared-state forks; trajectories generally differ
before the first challenge. Proposal-9-to-endpoint contrasts are therefore
descriptive. Repeated checkpoints estimate a periodic-challenge policy after
the first intervention, not independent treatment doses.

The message corpus contains public final responses, not private chain-of-thought.
Textual assumption language is a manipulation check. Source/AST changes,
evaluator outcomes, strict retention, descendant behavior, and costs provide
the substantive evidence. Ten-proposal follow-up gain includes all lineages;
the separate exact-ancestry files prevent it from being misreported as
descendant credit. Population-family conclusions reverse across reasonable
semantic representations, so the manuscript does not claim convergence.
