MECHANISM: Incremental TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.20 to 0.24 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18571599617004395.

INTENDED_EDIT: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.20 to 0.24.

EVIDENCE: Every verified increase from 0.04 through 0.20 preserved exactly 9,360 correct predictions and progressively lowered cross-entropy; the latest 0.16→0.20 step improved it from 0.18573341827392578 to 0.18571599617004395.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.20 * agreement)
=======
        confidence_scale = torch.exp(0.24 * agreement)
>>>>>>> REPLACE