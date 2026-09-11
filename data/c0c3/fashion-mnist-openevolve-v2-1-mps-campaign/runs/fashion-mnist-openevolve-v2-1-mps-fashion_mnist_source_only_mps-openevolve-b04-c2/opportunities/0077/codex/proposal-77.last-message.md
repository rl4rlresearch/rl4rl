MECHANISM: Prediction-invariant upper-side calibration bisection

HYPOTHESIS: Evaluation scale 1.17817215 will preserve all 9,279 argmax predictions and reduce validation cross-entropy below 0.20861771774291993.

INTENDED_EDIT: Change only the evaluation-time logit multiplier to the midpoint between the best-known scale 1.17817 and the worse 1.1781743 probe.

EVIDENCE: Scale 1.17817 achieved the best verified cross-entropy with 9,279 correct, while 1.1781743 was worse; bisecting their interval refines the calibration minimum without changing argmax classes.

<<<<<<< SEARCH
            logits = 1.17817 * logits
=======
            logits = 1.17817215 * logits
>>>>>>> REPLACE