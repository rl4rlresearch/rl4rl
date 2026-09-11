MECHANISM: One-ULP flip-ensemble power-mean refinement

HYPOTHESIS: The immediately lower float32 power-mean order will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

INTENDED_EDIT: Lower only the inference-time power-mean order by one float32 ULP.

EVIDENCE: The current order is best verified; a three-ULP increase worsened cross-entropy, while a 26-ULP decrease worsened it only slightly, making the untested immediate lower neighbor the most informative refinement.

<<<<<<< SEARCH
            power = 0.5831695556640625
=======
            power = 0.5831695199012756
>>>>>>> REPLACE