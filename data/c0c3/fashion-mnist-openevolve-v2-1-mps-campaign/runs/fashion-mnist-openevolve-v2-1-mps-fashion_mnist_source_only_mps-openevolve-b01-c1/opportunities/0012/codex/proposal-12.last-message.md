MECHANISM: Decision-preserving logit temperature sharpening

HYPOTHESIS: Scaling the successful 90% full-range/10% central ensemble logits by 1.05 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.272932.

INTENDED_EDIT: Apply a small positive logit scale at evaluation time without changing training, parameters, or predicted classes.

EVIDENCE: The current ensemble has the best verified correct count, and probability-space averaging worsened cross-entropy, indicating that softening its predictions is harmful; positive scaling preserves every argmax while testing modest sharpening.

<<<<<<< SEARCH
        return 0.9 * full_ensemble + 0.1 * central_ensemble
=======
        return 1.05 * (0.9 * full_ensemble + 0.1 * central_ensemble)
>>>>>>> REPLACE