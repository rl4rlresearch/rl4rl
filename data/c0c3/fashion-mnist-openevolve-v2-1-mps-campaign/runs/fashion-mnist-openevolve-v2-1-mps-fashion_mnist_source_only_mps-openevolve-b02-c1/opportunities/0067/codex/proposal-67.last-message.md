MECHANISM: Quadratic-fit post-ensemble temperature calibration

HYPOTHESIS: Temperature 0.912 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960927105.

INTENDED_EDIT: Change only the 12-view ensemble’s evaluation temperature from 0.91 to 0.912.

EVIDENCE: Temperatures 0.97, 0.925, and 0.91 produced cross-entropies of 0.1965603333, 0.1961161629, and 0.1960927105 with identical predictions; quadratic interpolation of these local measurements estimates the minimum near 0.9119.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.91
=======
        return (ensemble - math.log(12.0)) / 0.912
>>>>>>> REPLACE