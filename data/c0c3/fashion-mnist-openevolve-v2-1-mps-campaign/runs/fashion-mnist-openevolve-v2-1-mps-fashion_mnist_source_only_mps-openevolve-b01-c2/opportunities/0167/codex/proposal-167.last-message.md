MECHANISM: Late hard-target annealing

HYPOTHESIS: Linearly removing label smoothing during the final quarter of training, from the best verified inference fusion baseline, will exceed 9,331 correct predictions or retain 9,331 while lowering cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Restore the best verified complementary flip-fusion coefficients and anneal label smoothing from 0.02 to zero over the final 25% of optimization.

EVIDENCE: The 0.5085009765625 flipped-view weight achieved the best score twice, while fusion, calibration, EMA, probability fusion, and translation experiments failed to improve correctness; late target annealing tests an untried training-objective dimension while preserving the established representation-learning regime.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49149951171875 * logits + 0.50850048828125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
    paired_images = torch.cat(
=======
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * min(1.0, (1.0 - progress) / 0.25)
    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
=======
    view_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=label_smoothing,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    )
>>>>>>> REPLACE