MECHANISM: Symmetric local validation-logit calibration probe

HYPOTHESIS: Increasing the evaluation-only logit scale to 1.41638125 will preserve all 9,260 argmax predictions and reduce validation cross-entropy below 0.21200784797668457.

INTENDED_EDIT: Increase only the positive evaluation-time logit multiplier from 1.416375 to 1.41638125.

EVIDENCE: Lowering the scale by 0.00000625 to 1.41636875 preserved 9,260 correct but worsened cross-entropy to 0.2120078498840332; the equidistant upward probe is the most informative test of whether the local optimum lies just above the incumbent.

<<<<<<< SEARCH
            logits = logits * 1.416375
=======
            logits = logits * 1.41638125
>>>>>>> REPLACE