MECHANISM: Symmetric post-hoc logit-temperature calibration

HYPOTHESIS: Decreasing the best-known ensemble denominator by 0.5% will preserve all 9,192 predictions while reducing validation cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Retain the best-performing diagonal TTA weight and sharpen evaluation logits by dividing the ensemble sum by 99.5% of its exact normalization weight.

EVIDENCE: Increasing the denominator by 0.5% preserved correctness but worsened cross-entropy to 0.2380424545288086, directly motivating an accuracy-preserving probe in the opposite temperature direction.

<<<<<<< SEARCH
        return logit_sum / 15.7291015625
=======
        return logit_sum / 15.6504560546875
>>>>>>> REPLACE