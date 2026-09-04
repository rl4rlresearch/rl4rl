MECHANISM: Geometric flip-ensemble inference

HYPOTHESIS: Averaging paired-orientation logits will exceed 9,322 correct predictions by favoring cross-orientation consensus over a single view’s disproportionately confident probability.

INTENDED_EDIT: Replace probability-space flip ensembling with arithmetic logit averaging; training, parameters, and forward-pass count remain unchanged.

EVIDENCE: Paired-orientation supervision and flip-ensemble validation were beneficial, while adding Jensen–Shannon alignment timed out; logit averaging provides a computationally free consistency bias at inference.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE