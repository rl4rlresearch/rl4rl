MECHANISM: Incremental TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.16 to 0.20 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18573341827392578.

INTENDED_EDIT: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.16 to 0.20.

EVIDENCE: Each increase from 0.04 through 0.16 preserved exactly 9,360 correct predictions and progressively reduced cross-entropy; the latest 0.12→0.16 step improved it from 0.18575621643066406 to 0.18573341827392578.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.16 * agreement)
=======
        confidence_scale = torch.exp(0.20 * agreement)
>>>>>>> REPLACE