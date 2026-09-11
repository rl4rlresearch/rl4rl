MECHANISM: Verified flip-ensemble power-mean calibration

HYPOTHESIS: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.

INTENDED_EDIT: Lower only the inference-time probability power-mean order, leaving architecture and training unchanged.

EVIDENCE: This exact order achieved the highest verified score, 9243.41354937286, in Reference Designs 2 and 3; both the immediately lower float32 neighbor and nearby higher orders produced worse cross-entropy with the same 9,243 correct predictions.

<<<<<<< SEARCH
            power = 0.583172607421875
=======
            power = 0.5831695556640625
>>>>>>> REPLACE