MECHANISM: Confidence-boundary bisection

HYPOTHESIS: A 0.7471875 penultimate-step threshold will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120 by admitting only the higher-confidence subset of examples introduced at 0.746875.

INTENDED_EDIT: Lower the qualified penultimate-step confidence threshold from 0.75 to 0.7471875.

EVIDENCE: Threshold 0.7475 qualified at 85.03% with 17,360 recurrent steps, while 0.746875 admitted two additional exits but lost one correct validation prediction; bisecting that interval tests whether one candidate can exit safely.

<<<<<<< SEARCH
        return penultimate & (confidence >= 0.75)
=======
        return penultimate & (confidence >= 0.7471875)
>>>>>>> REPLACE