MECHANISM: Arithmetic probability TTA pooling

HYPOTHESIS: Averaging per-view class probabilities instead of logits will exceed 9,287 correct predictions by reducing the influence of overconfident errors from individual shifted or flipped views.

INTENDED_EDIT: Preserve the verified model, training procedure, center-view weight, and final calibration while replacing geometric logit aggregation with an arithmetic mixture of per-view probabilities.

EVIDENCE: Center-weight refinements plateaued at 9,287 correct and a larger weight lost one prediction, while the spatial-pooling redesign fell to 9,252; this motivates retaining the learned representation and testing the remaining load-bearing assumption—the form of multi-view aggregation.

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5578756246377452 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578756246377452
        return 1.22775 * pooled_logits
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        view_probabilities = view_logits.softmax(dim=-1)
        offset_probabilities = view_probabilities.mean(dim=1)
        pooled_probabilities = (
            1.5578756246377452 * offset_probabilities[0]
            + offset_probabilities[1:].sum(dim=0)
        ) / 5.5578756246377452
        return 1.22775 * pooled_probabilities.clamp_min(1.0e-8).log()
>>>>>>> REPLACE