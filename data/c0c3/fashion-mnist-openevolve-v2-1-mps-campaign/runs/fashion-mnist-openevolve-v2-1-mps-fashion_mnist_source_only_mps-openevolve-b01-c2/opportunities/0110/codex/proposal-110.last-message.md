MECHANISM: Late flip-consistency self-distillation

HYPOTHESIS: Ramping a small Jensen–Shannon-style consistency penalty between the two already-computed views will exceed 9,330 correct predictions by preventing late-stage flip disagreement without additional model forwards.

INTENDED_EDIT: Preserve the verified architecture, schedule, label smoothing, and calibration while adding a progress-ramped 0.05-weight consistency loss between original and flipped predictions.

EVIDENCE: The verified design reaches 9,330 correct, but its individual-view loss falls to zero at the final step; temperature scaling cannot change correctness, while higher-overhead augmentation and weight averaging did not finish. This reuses existing paired logits at negligible cost.

<<<<<<< SEARCH
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
>>>>>>> REPLACE

<<<<<<< SEARCH
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
=======
    original_probs = F.softmax(original_logits.detach(), dim=1)
    flipped_probs = F.softmax(flipped_logits.detach(), dim=1)
    consistency_target = 0.5 * (original_probs + flipped_probs)
    consistency_loss = 0.5 * (
        F.kl_div(
            F.log_softmax(original_logits, dim=1),
            consistency_target,
            reduction="batchmean",
        )
        + F.kl_div(
            F.log_softmax(flipped_logits, dim=1),
            consistency_target,
            reduction="batchmean",
        )
    )
    return (
        ensemble_weight * ensemble_loss
        + (1.0 - ensemble_weight) * view_loss
        + 0.05 * progress * consistency_loss
    )
>>>>>>> REPLACE