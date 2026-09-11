MECHANISM: Evidence-fitted post-ensemble temperature calibration

HYPOTHESIS: Temperature 0.91 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196116.

INTENDED_EDIT: Sharpen the unchanged 12-view ensemble by changing its evaluation temperature from 0.925 to 0.91.

EVIDENCE: Cross-entropy improved from 0.197943 at temperature 1.03 to 0.196560 at 0.97 and 0.196116 at 0.925 with identical predictions; a quadratic interpolation of these measured points places the local minimum near 0.91.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.925
=======
        return (ensemble - math.log(12.0)) / 0.91
>>>>>>> REPLACE