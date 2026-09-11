MECHANISM: Conservative power-mean boundary refinement

HYPOTHESIS: A 0.5831787109375-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904550170898437.

INTENDED_EDIT: Lower only the inference-time flip-ensemble power-mean order from 0.58359375 to 0.5831787109375.

EVIDENCE: Order 0.583203125 retained 9,243 correct, while 0.5830078125 lost one prediction; 0.5831787109375 is a conservative refinement toward the lower-cross-entropy boundary, and the intervening timeout supplied no contradictory accuracy evidence.

<<<<<<< SEARCH
            power = 0.58359375
=======
            power = 0.5831787109375
>>>>>>> REPLACE