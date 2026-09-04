# Paper 6 reproducibility notes

This directory contains the deterministic analysis and manuscript builder for:

**Question the Premise, Pay the Price: Assumption-Challenge Prompts in
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

## Reproduce the analysis

From the repository root:

```bash
python papers/aiscik2026/paper6/analysis.py
```

The script validates the run roster, horizons, condition mapping, intervention
schedule, and prompt placement before writing outputs under `derived/`. It uses
fixed bootstrap seed `20260903`. Numerical CSV and JSON files are deterministic;
rendered PNG metadata may vary by environment.

Dependencies are Python 3.11+ and Matplotlib. The PDF builder additionally
requires ReportLab and pypdf is useful for verification.

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
- three rendered figures.

`qualitative_audit.md` records the human-readable trace analysis used in the
paper. `section_expansion.md` is an author-facing evidence bank with additional
results and examples for every manuscript section.

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
the substantive evidence.
