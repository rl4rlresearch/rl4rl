MECHANISM: Evaluation-only temperature sharpening

HYPOTHESIS: Sharpening the fused validation logits with temperature 0.90 will preserve all 9,328 argmax predictions while lowering cross-entropy below 0.201899, producing a strict validation-score improvement.

INTENDED_EDIT: Divide the confidence-adaptive fused log-probabilities by 0.90 during evaluation; training remains unchanged.

EVIDENCE: Constant 0.02 label smoothing achieved the best count, 9,328, but raised cross-entropy to 0.201899 versus roughly 0.1922 without smoothing; late smoothing decay lowered cross-entropy to 0.189546. This indicates underconfidence that post-hoc sharpening can address without changing predicted classes.

<<<<<<< SEARCH
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
=======
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.90
>>>>>>> REPLACE