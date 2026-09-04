MECHANISM: Mild post-ensemble logit sharpening

HYPOTHESIS: Temperature 0.97 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Divide the existing sequential 12-weight TTA ensemble logits by 0.97 without changing training, parameters, or evaluation workload.

EVIDENCE: Temperature 1.03 preserved 9,286 correct but worsened cross-entropy from 0.197145 to 0.197943, directly indicating that the locally favorable calibration direction is sharpening rather than softening.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 0.97
>>>>>>> REPLACE