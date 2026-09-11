MECHANISM: Accuracy-invariant logit temperature refinement

HYPOTHESIS: Decreasing the evaluation-time logit scale from 1.104 to 1.1035 will preserve all 9,360 argmax predictions while reducing validation cross-entropy below 0.18600157318115235.

INTENDED_EDIT: Lower only the positive evaluation-time logit multiplier, leaving training, TTA views, center weighting, normalization, and class rankings unchanged.

EVIDENCE: Increasing the scale to 1.1045 preserved 9,360 correct but worsened cross-entropy to 0.18600544357299806, indicating that the locally favorable calibration direction is downward.

<<<<<<< SEARCH
        return 1.104 * logits
=======
        return 1.1035 * logits
>>>>>>> REPLACE