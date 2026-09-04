MECHANISM: Confidence-boundary bisection

HYPOTHESIS: A 0.746875 penultimate-step threshold will retain validation_accuracy >= 0.85 and may reduce total_inference_macs below 746,601,120 by admitting a safe subset of examples excluded at 0.7475.

INTENDED_EDIT: Lower the current penultimate-step confidence threshold from 0.90 to 0.746875.

EVIDENCE: The 0.7475 threshold qualified at 85.03% and 746,601,120 MACs, while 0.74625 admitted three additional exits but missed the accuracy requirement by one example; bisecting their interval directly probes the remaining boundary.

<<<<<<< SEARCH
        return penultimate & (confidence >= 0.90)
=======
        return penultimate & (confidence >= 0.746875)
>>>>>>> REPLACE