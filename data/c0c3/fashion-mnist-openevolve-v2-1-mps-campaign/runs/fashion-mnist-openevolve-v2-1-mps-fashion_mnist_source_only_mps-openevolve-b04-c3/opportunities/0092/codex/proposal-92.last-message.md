MECHANISM: Verified flip-ensemble power-mean calibration

HYPOTHESIS: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions while reducing validation cross-entropy below the current 0.20904550170898437.

INTENDED_EDIT: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.

EVIDENCE: Reference Designs 2 and 3 achieved the highest verified validation score, 9243.41354937286, with this exact order; neighboring tested orders were no better while retaining the same correct count.

<<<<<<< SEARCH
            power = 0.583203125
=======
            power = 0.5831695556640625
>>>>>>> REPLACE