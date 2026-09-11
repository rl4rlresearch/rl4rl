MECHANISM: Incremental TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.28 to 0.32 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18569724464416504.

INTENDED_EDIT: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.28 to 0.32.

EVIDENCE: Every verified 0.04 coefficient increase through 0.28 preserved exactly 9,360 correct predictions and lowered cross-entropy; the latest 0.24→0.28 step improved it from 0.1857039375305176 to 0.18569724464416504, supporting one further equal-sized step despite diminishing gains.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.28 * agreement)
=======
        confidence_scale = torch.exp(0.32 * agreement)
>>>>>>> REPLACE