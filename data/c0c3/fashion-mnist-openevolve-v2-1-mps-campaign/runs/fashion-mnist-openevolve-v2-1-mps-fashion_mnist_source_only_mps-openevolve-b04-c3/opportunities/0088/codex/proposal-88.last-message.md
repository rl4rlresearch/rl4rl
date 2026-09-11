MECHANISM: Verified flip-ensemble power-mean calibration

HYPOTHESIS: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions and reduce validation cross-entropy from 0.209074520111084 toward the best verified 0.20904547996520997.

INTENDED_EDIT: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.

EVIDENCE: Reference Design 3 achieved the highest verified validation score, 9243.41354937286, with this exact order; nearby verified orders were no better, while repeat failures were timeouts or unverifiable rather than contradictory results.

<<<<<<< SEARCH
            power = 0.625
=======
            power = 0.5831695556640625
>>>>>>> REPLACE