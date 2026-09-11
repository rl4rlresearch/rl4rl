MECHANISM: Low-weight hardest-rival margin supervision

HYPOTHESIS: Adding a 0.05-weight margin penalty to the ensemble logits will increase validation correctness beyond 9,330 by directly correcting low-margin decisions while preserving the verified training procedure and calibrated evaluation scale.

INTENDED_EDIT: Supplement the existing smoothed cross-entropy mixture with a hinge penalty requiring the true-class ensemble logit to exceed the strongest competing logit by 0.5.

EVIDENCE: The current design already achieves the best verified calibration at 9,330 correct; further temperature scaling cannot change its argmax predictions, while prior hard-label experiments timed out without contrary accuracy evidence. A lightweight margin term targets the primary correctness metric without additional model forwards.

<<<<<<< SEARCH
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
=======
    base_loss = (
        ensemble_weight * ensemble_loss
        + (1.0 - ensemble_weight) * view_loss
    )
    correct_logits = ensemble_logits.gather(1, labels[:, None]).squeeze(1)
    correct_mask = F.one_hot(labels, num_classes=10).bool()
    strongest_other = ensemble_logits.masked_fill(
        correct_mask,
        float("-inf"),
    ).amax(dim=1)
    margin_loss = F.relu(
        0.5 - correct_logits + strongest_other
    ).mean()
    return base_loss + 0.05 * margin_loss
>>>>>>> REPLACE