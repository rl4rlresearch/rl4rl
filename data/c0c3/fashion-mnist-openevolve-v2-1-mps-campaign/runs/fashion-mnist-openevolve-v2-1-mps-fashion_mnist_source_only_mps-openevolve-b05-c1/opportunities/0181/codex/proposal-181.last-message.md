MECHANISM: Incremental TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.24 to 0.28 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.1857039375305176.

INTENDED_EDIT: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.24 to 0.28.

EVIDENCE: Every verified increase from 0.04 through 0.24 preserved exactly 9,360 correct predictions and progressively lowered cross-entropy; the latest 0.20→0.24 step improved it from 0.18571599617004395 to 0.1857039375305176.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.24 * agreement)
=======
        confidence_scale = torch.exp(0.28 * agreement)
>>>>>>> REPLACE