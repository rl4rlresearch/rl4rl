MECHANISM: Nearest-point quadratic logit calibration

HYPOTHESIS: An evaluation scale of 1.167286 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258541.

INTENDED_EDIT: Restore the best validated center-compensated TTA weights and replace the evaluation multiplier with the quadratic vertex estimated from the three nearest calibration results.

EVIDENCE: Scales 1.1645, 1.16727, and 1.1682 yielded cross-entropies 0.207259576, 0.207258541, and 0.207258652; their fitted minimum is approximately 1.167286, while positive global scaling preserves class rankings.

<<<<<<< SEARCH
        logits = 0.3634375 * self._flip_average(images)
=======
        logits = 0.3640625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                        weight = 0.10875
=======
                        weight = 0.1084375
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.253 * logits
=======
        return 1.167286 * logits
>>>>>>> REPLACE