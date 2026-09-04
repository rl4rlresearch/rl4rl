MECHANISM: Validation-aligned flip-loss curriculum

HYPOTHESIS: Annealing from equal ensemble/view supervision to pure ensemble supervision, while preserving the successful 75% average ensemble weight, will exceed 9,325 correct predictions.

INTENDED_EDIT: Linearly increase the ensemble-loss weight from 0.5 to 1.0 across training and decrease the per-view weight correspondingly.

EVIDENCE: Static 75% ensemble weighting reported 9,325 correct versus 9,312 for equal weighting, while earlier per-view supervision improved 9,261 to 9,276; this schedule retains per-view guidance early and emphasizes the validation-matched ensemble late.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
=======
) -> torch.Tensor:
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    return 0.5 * (ensemble_loss + view_loss)
=======
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
>>>>>>> REPLACE