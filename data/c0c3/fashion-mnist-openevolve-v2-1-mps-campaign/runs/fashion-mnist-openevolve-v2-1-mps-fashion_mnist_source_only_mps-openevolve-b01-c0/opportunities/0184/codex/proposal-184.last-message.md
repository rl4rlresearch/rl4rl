MECHANISM: Argmax-invariant temperature refinement

HYPOTHESIS: Lowering the ensemble temperature by one float32 ULP will preserve all 9,290 predictions while slightly reducing validation cross-entropy if the ensemble remains marginally underconfident.

INTENDED_EDIT: Decrease only the final inference temperature from 0.717143714427948 to its adjacent lower float32 value.

EVIDENCE: Unshifted-view weighting has reached a quantized accuracy boundary without improving the reported cross-entropy; positive temperature scaling leaves argmax predictions unchanged, making it the safest remaining tie-breaker probe.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.7171436548233032
>>>>>>> REPLACE