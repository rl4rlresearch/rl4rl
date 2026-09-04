MECHANISM: Verified flip-ensemble power-mean calibration

HYPOTHESIS: Using the verified 0.5831695556640625 order will retain 9,243 correct predictions and reduce validation cross-entropy from 0.20904548225402833 to approximately 0.20904547996520997.

INTENDED_EDIT: Replace only the inference-time probability power-mean order with the best verified value.

EVIDENCE: Reference Design 2 used this exact order with the otherwise identical implementation and achieved the highest available validation score, 9243.41354937286.

<<<<<<< SEARCH
            power = 0.583172607421875
=======
            power = 0.5831695556640625
>>>>>>> REPLACE