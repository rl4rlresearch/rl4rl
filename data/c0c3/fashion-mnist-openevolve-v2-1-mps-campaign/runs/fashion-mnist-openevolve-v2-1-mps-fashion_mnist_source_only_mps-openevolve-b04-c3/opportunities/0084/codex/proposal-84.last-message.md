MECHANISM: Verified flip-ensemble power-mean calibration

HYPOTHESIS: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions and reduce validation cross-entropy below the current 0.2090745201.

INTENDED_EDIT: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.

EVIDENCE: Reference Design 3 used this exact order with the otherwise identical implementation and achieved the highest verified score, 9243.41354937286, with cross-entropy 0.20904547996520997.

<<<<<<< SEARCH
            power = 0.625
=======
            power = 0.5831695556640625
>>>>>>> REPLACE