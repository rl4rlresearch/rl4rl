MECHANISM: Mean-preserving ensemble-loss curriculum

HYPOTHESIS: Linearly increasing ensemble supervision from 0.60 to 0.90 while preserving an average weight of 0.75 will exceed 9,236 correct predictions by prioritizing stable per-view feature learning early and deployed-ensemble alignment late.

INTENDED_EDIT: Replace the static 75% ensemble-loss weight with a training-progress curriculum from 60% to 90%, leaving the architecture, views, optimizer, and average loss weighting unchanged.

EVIDENCE: Static 0.75 achieved the best result at 9,236 correct, while both lower 0.70 and higher 0.80/0.875 weights regressed; preserving the successful mean while changing only its temporal allocation directly tests whether those competing supervision needs are stage-dependent.

<<<<<<< SEARCH
    del step, total_steps
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE

<<<<<<< SEARCH
    return 0.25 * individual_loss + 0.75 * ensemble_loss
=======
    ensemble_weight = 0.60 + 0.30 * progress
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
>>>>>>> REPLACE