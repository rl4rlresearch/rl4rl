MECHANISM: Evidence-fitted post-ensemble temperature sharpening

HYPOTHESIS: Temperature 0.91 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196116.

INTENDED_EDIT: Change the unchanged 12-view ensemble’s evaluation temperature from 0.925 to 0.91.

EVIDENCE: Cross-entropy improved monotonically from 0.197943 at temperature 1.03 to 0.196560 at 0.97 and 0.196116 at 0.925 with identical predictions; quadratic interpolation places the estimated minimum near 0.91, and the previous 0.91 attempt timed out without testing the hypothesis.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.925
=======
        return (ensemble - math.log(12.0)) / 0.91
>>>>>>> REPLACE