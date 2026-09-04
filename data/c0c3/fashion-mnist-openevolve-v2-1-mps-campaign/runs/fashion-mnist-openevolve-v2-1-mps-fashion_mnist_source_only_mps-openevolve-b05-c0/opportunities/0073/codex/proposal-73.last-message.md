MECHANISM: Quadratic-guided inference-logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.225 will preserve all argmax predictions while reducing validation cross-entropy below 0.219309.

INTENDED_EDIT: Increase only the evaluation-time logit multiplier from 1.2 to 1.225.

EVIDENCE: Cross-entropy fell from 0.241946 at 1.0× to 0.226035 at 1.1× and 0.219309 at 1.2× without changing predictions; a quadratic fit to those measurements places the approximate minimum near 1.223×.

<<<<<<< SEARCH
            logits = logits * 1.2
=======
            logits = logits * 1.225
>>>>>>> REPLACE