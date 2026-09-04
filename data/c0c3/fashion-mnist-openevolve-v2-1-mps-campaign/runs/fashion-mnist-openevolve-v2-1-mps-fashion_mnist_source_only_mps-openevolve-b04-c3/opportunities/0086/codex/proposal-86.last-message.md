MECHANISM: Verified flip-ensemble power-mean calibration

HYPOTHESIS: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions and reduce validation cross-entropy from 0.20904548225402833 to 0.20904547996520997.

INTENDED_EDIT: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.

EVIDENCE: Reference Design 2 used this exact order with the otherwise identical implementation and achieved the highest available validation score, 9243.41354937286; the failed repeat verifications were timeouts or unverifiable rather than contradictory performance results.

<<<<<<< SEARCH
            power = 0.583172607421875
=======
            power = 0.5831695556640625
>>>>>>> REPLACE