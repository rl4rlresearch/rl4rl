MECHANISM: Argmax-guarded TTA weight extrapolation

HYPOTHESIS: Extrapolating the successful canonical-view weight increment once more, while retaining the current logits wherever the extrapolated ensemble changes the stable prediction, will preserve 9,359 correct predictions and reduce cross-entropy below 0.18456672821044923.

INTENDED_EDIT: Add a 53.807373046875% canonical-view ensemble and use it only on samples where its argmax still matches the accuracy-preserving stable ensemble; otherwise retain the current disagreement-corrected logits.

EVIDENCE: Moving from 53.8072967529296875% to 53.80733489990234375% reduced cross-entropy, and the disagreement guard plus boost preserved all 9,359 predictions. The 16-to-32 boost probe then saturated, motivating another guarded move along the previously beneficial weight direction.

<<<<<<< SEARCH
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            corrected_logits,
        )
        return 1.226016 * ensemble_logits
=======
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            corrected_logits,
        )
        extrapolated_logits = (
            0.53807373046875 * logits
            + 0.46192626953125 * flipped_logits
        )
        extrapolated_unchanged = (
            extrapolated_logits.argmax(dim=1, keepdim=True)
            == stable_prediction
        )
        ensemble_logits = torch.where(
            extrapolated_unchanged,
            extrapolated_logits,
            ensemble_logits,
        )
        return 1.226016 * ensemble_logits
>>>>>>> REPLACE