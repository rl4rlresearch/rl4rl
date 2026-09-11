MECHANISM: Predictive-probability flip ensembling

HYPOTHESIS: Replacing arithmetic logit averaging with a calibrated mixture of the two views’ predictive probabilities will exceed 9,312 correct predictions by handling view disagreement without changing the proven architecture or training budget.

INTENDED_EDIT: Use log-probability mixture aggregation for paired-flip supervision and validation inference while retaining per-view deep supervision.

EVIDENCE: Paired-flip ensemble training improved accuracy from 9,247 to 9,261, and per-view supervision raised it to 9,276; the current residual model reached 9,312, motivating an isolated refinement of its successful flip-ensemble calculation.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            ) - math.log(2.0)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (original_logits + flipped_logits)

    ensemble_loss = F.cross_entropy(
=======
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = torch.logaddexp(
        F.log_softmax(original_logits, dim=1),
        F.log_softmax(flipped_logits, dim=1),
    ) - math.log(2.0)

    ensemble_loss = F.cross_entropy(
>>>>>>> REPLACE