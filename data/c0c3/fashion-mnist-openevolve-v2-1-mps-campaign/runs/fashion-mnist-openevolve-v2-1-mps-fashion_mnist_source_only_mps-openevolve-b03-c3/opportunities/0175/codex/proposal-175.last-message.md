MECHANISM: Accuracy-preserving continuation of global logit recalibration

HYPOTHESIS: Extending Reference Design 1’s successful scale reduction from 1.252 to 1.251 will retain exactly 9,284 correct predictions while lowering validation cross-entropy below 0.208112982.

INTENDED_EDIT: Restore Reference Design 1’s validated TTA weights and decrease only its positive evaluation logit multiplier by another 0.001.

EVIDENCE: Reducing the multiplier from 1.253 to 1.252 preserved all 9,284 predictions and reduced cross-entropy from 0.208132217 to 0.208112982; another small positive-scale reduction cannot change argmax rankings and directly tests whether calibration continues improving.

<<<<<<< SEARCH
        logits = 0.3625 * self._flip_average(images)
=======
        logits = 0.3640625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.07000
=======
                    if delta_x == 0:
                        weight = 0.1084375
                    elif delta_y == 0:
                        weight = 0.06953125
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.253 * logits
=======
        return 1.251 * logits
>>>>>>> REPLACE