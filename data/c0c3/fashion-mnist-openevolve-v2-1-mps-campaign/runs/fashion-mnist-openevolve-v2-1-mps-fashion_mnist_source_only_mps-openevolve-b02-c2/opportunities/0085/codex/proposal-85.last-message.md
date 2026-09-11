MECHANISM: Disagreement-conditioned flip calibration

HYPOTHESIS: Softening logits only when the original and flipped views predict different classes will preserve all 9,322 ensemble predictions while lowering validation cross-entropy on these uncertainty-signaling cases.

INTENDED_EDIT: Keep training and arithmetic flip ensembling unchanged, but apply temperature 1.10 only to validation samples whose two views disagree.

EVIDENCE: Arithmetic probability ensembling achieved lower cross-entropy than geometric ensembling at the same 9,320 correct predictions, indicating that handling flip-view uncertainty affects the tie-breaker; positive per-sample temperature scaling cannot change the ensemble argmax.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        ensemble_logits = (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        )
        views_disagree = logits.argmax(dim=1) != flipped_logits.argmax(dim=1)
        temperature = 1.0 + 0.10 * views_disagree.to(ensemble_logits.dtype)
        return ensemble_logits / temperature.unsqueeze(1)
>>>>>>> REPLACE