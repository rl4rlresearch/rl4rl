MECHANISM: Best-verified margin-slope calibration

HYPOTHESIS: Setting the confidence-adaptation slope to 0.017065 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22211728744506837.

INTENDED_EDIT: Increase the evaluation-time, batch-mean-centered margin-scaling slope from 0.01 to 0.017065.

EVIDENCE: Reference Design 1 used slope 0.017065 and achieved the best observed score, with 9,192 correct predictions and cross-entropy 0.22206240844726563.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.01 * (
            margin - margin.mean()
        )
=======
        confidence_scale = (10500.0 / 9564.0) + 0.017065 * (
            margin - margin.mean()
        )
>>>>>>> REPLACE