MECHANISM: Evidence-fitted TTA-agreement temperature calibration

HYPOTHESIS: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569591064453125.

INTENDED_EDIT: Reduce only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.32 to the estimated optimum of 0.31.

EVIDENCE: Successive 0.04 coefficient increases showed nearly constant diminishing gains, fitting a cross-entropy minimum near 0.31; two prior 0.31 runs timed out but produced no contrary validation evidence, and positive scalar scaling cannot change argmax predictions.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.32 * agreement)
=======
        confidence_scale = torch.exp(0.31 * agreement)
>>>>>>> REPLACE