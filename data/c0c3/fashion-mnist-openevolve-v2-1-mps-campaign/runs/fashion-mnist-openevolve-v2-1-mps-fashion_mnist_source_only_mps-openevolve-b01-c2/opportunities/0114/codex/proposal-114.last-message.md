MECHANISM: Canonical-view-biased flip ensemble

HYPOTHESIS: Weighting the original validation view 0.55 and its horizontal flip 0.45 will correct at least one borderline flip disagreement and exceed 9,330 correct predictions without materially degrading calibrated cross-entropy.

INTENDED_EDIT: Replace equal evaluation-time logit fusion with a slight original-view bias while retaining the verified 1.184 temperature.

EVIDENCE: The current equal-weight ensemble achieves 9,330 correct, while temperature scaling is already optimized and cannot alter argmax predictions; the attempted probability-space fusion did not finish, leaving low-cost fusion changes as an untested route to higher correctness.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.55 * logits + 0.45 * flipped_logits)
>>>>>>> REPLACE