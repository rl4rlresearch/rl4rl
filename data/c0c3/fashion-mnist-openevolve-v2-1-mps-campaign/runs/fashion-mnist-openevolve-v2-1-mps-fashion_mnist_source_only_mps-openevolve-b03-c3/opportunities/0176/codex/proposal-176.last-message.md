MECHANISM: Accuracy-preserving continuation of global logit recalibration

HYPOTHESIS: Extending Reference Design 3’s scale reduction from 1.251 to 1.250 will preserve exactly 9,284 correct predictions while lowering validation cross-entropy below 0.208093939.

INTENDED_EDIT: Restore Reference Design 3’s validated TTA weights and reduce only its evaluation logit multiplier by 0.001.

EVIDENCE: Consecutive reductions from 1.253 to 1.252 and then 1.251 preserved all 9,284 predictions while monotonically lowering cross-entropy from 0.208132217 to 0.208112982 and 0.208093939; positive global scaling leaves argmax rankings unchanged.

<<<<<<< SEARCH
        logits = 0.361875 * self._flip_average(images)
=======
        logits = 0.3640625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.0703125
=======
                    if delta_x == 0:
                        weight = 0.1084375
                    elif delta_y == 0:
                        weight = 0.06953125
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.253 * logits
=======
        return 1.250 * logits
>>>>>>> REPLACE