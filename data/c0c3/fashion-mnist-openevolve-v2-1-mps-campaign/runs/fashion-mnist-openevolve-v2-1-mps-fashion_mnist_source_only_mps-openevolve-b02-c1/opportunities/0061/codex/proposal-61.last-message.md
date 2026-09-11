MECHANISM: Evidence-guided post-ensemble temperature calibration

HYPOTHESIS: Temperature 0.925 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.

INTENDED_EDIT: Sharpen the unchanged sequential TTA ensemble by dividing its logits by 0.925 instead of 0.97.

EVIDENCE: Temperature 1.03 worsened cross-entropy to 0.197943, while 0.97 improved it to 0.196560 with identical predictions; these measured points indicate the local optimum lies further in the sharpening direction.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.97
=======
        return (ensemble - math.log(12.0)) / 0.925
>>>>>>> REPLACE