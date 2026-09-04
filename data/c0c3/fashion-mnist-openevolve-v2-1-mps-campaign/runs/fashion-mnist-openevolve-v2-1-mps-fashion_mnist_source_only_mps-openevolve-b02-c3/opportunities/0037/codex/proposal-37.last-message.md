MECHANISM: Quadratic-fit posterior temperature calibration

HYPOTHESIS: Setting the ensemble temperature to 0.753 will preserve all 9,315 argmax predictions and reduce validation cross-entropy below 0.1913601.

INTENDED_EDIT: Change only the positive inference temperature from 0.75 to 0.753.

EVIDENCE: Temperatures 0.78, 0.76, and 0.75 preserved identical predictions while cross-entropy reached 0.1915621, 0.1913707, and 0.1913601; a local quadratic fit places the minimum near 0.753.

<<<<<<< SEARCH
        ).log() / 0.75
=======
        ).log() / 0.753
>>>>>>> REPLACE