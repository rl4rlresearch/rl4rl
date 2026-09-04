MECHANISM: Quadratically tuned margin-adaptive calibration

HYPOTHESIS: A slope of 0.0175 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22207223510742188, because the gains from slopes 0.01 and 0.02 imply a shallow optimum near 0.017.

INTENDED_EDIT: Replace uniform fused calibration with max-centered ensemble normalization and batch-mean-centered margin scaling at slope 0.0175.

EVIDENCE: Margin adaptation improved cross-entropy from roughly 0.22237433 at slope 0 to 0.22211729 at 0.01 and 0.22207224 at 0.02 without changing correct predictions; quadratic interpolation of those three observations places the estimated optimum near 0.0171.

<<<<<<< SEARCH
        calibrated_logits = logit_sum * (
            1.0978671312332153 / 13.2124453125
        )
        return calibrated_logits - calibrated_logits.amax(
            dim=1, keepdim=True
        )
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        top_two = normalized_logits.topk(2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        confidence_scale = (10500.0 / 9564.0) + 0.0175 * (
            margin - margin.mean()
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
        return normalized_logits * confidence_scale
>>>>>>> REPLACE