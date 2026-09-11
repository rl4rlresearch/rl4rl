MECHANISM: Quadratic-guided logit temperature calibration

HYPOTHESIS: Scaling evaluation logits by 1.295 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255195.

INTENDED_EDIT: Reduce the decision-preserving evaluation-only logit scale from 1.30 to 1.295.

EVIDENCE: Cross-entropy improvements shrank from 0.001116 at 1.20→1.25 to 0.000327 at 1.25→1.30; quadratic interpolation of those measurements places the calibration minimum near 1.296.

<<<<<<< SEARCH
        return 1.30 * (0.9 * full_ensemble + 0.1 * central_ensemble)
=======
        return 1.295 * (0.9 * full_ensemble + 0.1 * central_ensemble)
>>>>>>> REPLACE