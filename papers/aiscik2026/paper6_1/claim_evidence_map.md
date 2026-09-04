# Claim-to-evidence map

This map identifies the frozen analytic record behind each headline statement.
All paths are relative to this directory.

| Manuscript claim | Direct evidence | Qualification |
|---|---|---|
| 52 trajectories, 6,080 proposals, 304 challenged checkpoints, 608 matched checkpoint opportunities | `derived/integrity.json`; `derived/proposal_records.csv`; `derived/checkpoint_pairs.csv` | Horizons are task-specific: addition 80, Fashion-MNIST 200, nanoGPT 40. |
| Challenged proposals depart lexically from their own prior trajectory in all tasks | `derived/checkpoint_effects.csv`, `metric=lexical_novelty`; `derived/block_checkpoint_effects.csv`; `derived/robustness_summary.json` | Lexical departure is not historical scientific novelty. All 13 blocks have positive local effects. |
| Candidate source changes increase on Fashion-MNIST and nanoGPT but not clearly on addition | `derived/checkpoint_effects.csv`, `metric in {source_novelty,ast_distance,changed_lines}`; `derived/source_missingness.csv` | Finite local source contrasts are 77/80, 169/200, and 24/24 by task; feasibility retains full denominators. |
| Challenges cost 2.8k–4.8k more output tokens locally | `derived/checkpoint_effects.csv`, `metric=output_tokens`; `derived/block_checkpoint_effects.csv` | All 13 blocks have positive local output-token effects. Dollar cost is not claimed. |
| Immediate retention falls | `derived/checkpoint_effects.csv`, `metric=retained`; `derived/robustness_summary.json` | Negative in 11 of 12 nonzero block estimates; strict task-specific evaluators define retention. |
| Ten-proposal cycle gain favors challenged trajectories | `derived/cycle_gain_effects.csv`; `derived/cycle_gain_pairs.csv` | Associated outcome under a repeated policy, not a randomized causal endpoint effect. |
| Ten-proposal windows can improve after the challenged checkpoint | `derived/cycle_gain_effects.csv`; `derived/cycle_gain_pairs.csv` | Includes every lineage selected in the next nine proposals and is not descendant credit. |
| Productive alternatives can become executable descendant branches | `derived/lineage_descendant_cycles.csv`; `derived/lineage_descendant_summary.csv`; `qualitative_audit.md`, cases A1/F1/N1 | Exact ancestry shows branch formation is uncommon and descendant-only gain is task-dependent; public rationales are not private reasoning. |
| Population conclusions reverse across reasonable family representations | `derived/population_dispersion.csv`; `derived/population_measure_sensitivity.csv`; `derived/population_measure_sensitivity_checkpoints.csv` | Full-rationale tags suggest concentration, while mechanism-only and primary-family coding show greater challenged dispersion in every task; no convergence claim is retained. |
| Portfolio memory has no stable interaction pattern | memory-stratified rows in `derived/checkpoint_effects.csv` and `derived/endpoint_effects.csv` | Five blocks on addition/Fashion-MNIST and three on nanoGPT are underpowered for a general interaction conclusion. |
| Pre-intervention transitions do not show the intervention's source/token pattern | `derived/preintervention_placebos.csv`; `derived/preintervention_placebo_effects.csv` | A falsification check, not proof of random assignment. |
| Leave-one-block-out signs are stable for lexical departure, source departure, retention, and output cost | `derived/leave_one_block_out.csv`; `derived/robustness_summary.json` | Stability applies within each task and does not eliminate state imbalance. |
