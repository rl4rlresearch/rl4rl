MECHANISM: Evidence-fitted TTA-agreement temperature calibration

HYPOTHESIS: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569591064453125.

INTENDED_EDIT: Reduce only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.32 to 0.31.

EVIDENCE: Successive coefficient increases produced diminishing cross-entropy gains, placing the estimated minimum near 0.31; the previous 0.31 verification timed out and supplied no contrary validation evidence.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.32 * agreement)
=======
        confidence_scale = torch.exp(0.31 * agreement)
>>>>>>> REPLACE