MECHANISM: Local quadratic temperature calibration

HYPOTHESIS: Temperature 0.912028 will preserve all 9,286 predictions while reducing validation cross-entropy below 0.19609205169677735.

INTENDED_EDIT: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.912028.

EVIDENCE: Temperatures 0.91, 0.912, and 0.91213 produced cross-entropies of 0.1960927105, 0.1960920517, and 0.1960920532 with identical predictions; quadratic interpolation of these closest measurements estimates the minimum near 0.912028.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.912
=======
        return (ensemble - math.log(12.0)) / 0.912028
>>>>>>> REPLACE