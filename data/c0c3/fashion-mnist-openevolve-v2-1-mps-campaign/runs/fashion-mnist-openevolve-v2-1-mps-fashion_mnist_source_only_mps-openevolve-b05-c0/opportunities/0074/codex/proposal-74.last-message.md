MECHANISM: Quadratic-fit inference calibration

HYPOTHESIS: Scaling evaluation logits by 1.223 will retain 9,258 correct predictions while reducing validation cross-entropy below 0.219309.

INTENDED_EDIT: Increase only the evaluation-time logit multiplier from 1.2 to 1.223.

EVIDENCE: Cross-entropy decreased at every tested scale—0.241946 at 1.0, 0.226035 at 1.1, and 0.219309 at 1.2—without changing predictions; a quadratic fit places the estimated minimum near 1.223, while the previous 1.225 run timed out without contradictory calibration evidence.

<<<<<<< SEARCH
            logits = logits * 1.2
=======
            logits = logits * 1.223
>>>>>>> REPLACE