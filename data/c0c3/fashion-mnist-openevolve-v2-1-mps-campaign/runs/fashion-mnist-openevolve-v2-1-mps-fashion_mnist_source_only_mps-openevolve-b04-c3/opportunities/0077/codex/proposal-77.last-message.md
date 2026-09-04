MECHANISM: Conservative flip-ensemble power-mean boundary refinement

HYPOTHESIS: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.5831787109375 to 0.583172607421875.

EVIDENCE: Order 0.5831787109375 achieved the best verified result with 9,243 correct, while 0.5830078125 lost one prediction; the closer 0.583172607421875 probe previously timed out and therefore remains the most informative unresolved refinement.

<<<<<<< SEARCH
            power = 0.5831787109375
=======
            power = 0.583172607421875
>>>>>>> REPLACE