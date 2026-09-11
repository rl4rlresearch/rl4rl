MECHANISM: Fine-grained post-ensemble temperature calibration

HYPOTHESIS: Temperature 0.91213 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960920517.

INTENDED_EDIT: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.91213.

EVIDENCE: Temperatures 0.91, 0.912, and 0.925 produced cross-entropies of 0.1960927105, 0.1960920517, and 0.1961161629 with identical predictions; quadratic interpolation of these local measurements estimates the minimum near 0.91213.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.912
=======
        return (ensemble - math.log(12.0)) / 0.91213
>>>>>>> REPLACE