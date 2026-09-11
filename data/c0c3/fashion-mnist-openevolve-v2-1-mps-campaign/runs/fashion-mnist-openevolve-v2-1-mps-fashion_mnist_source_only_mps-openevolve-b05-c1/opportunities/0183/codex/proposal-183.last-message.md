MECHANISM: Evidence-fitted TTA-agreement temperature calibration

HYPOTHESIS: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18569591064453125.

INTENDED_EDIT: Reduce only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.32 to 0.31.

EVIDENCE: Cross-entropy improvements from successive 0.04 increases diminished consistently by about 5.36e-6 per step; fitting that trend places the minimum near 0.31, while positive scalar calibration cannot alter argmax predictions.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.32 * agreement)
=======
        confidence_scale = torch.exp(0.31 * agreement)
>>>>>>> REPLACE