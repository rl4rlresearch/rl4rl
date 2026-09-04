MECHANISM: Conservative power-mean boundary refinement

HYPOTHESIS: A 0.583154296875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.

INTENDED_EDIT: Lower only the inference-time flip-ensemble power-mean order from 0.583203125 to 0.583154296875.

EVIDENCE: Order 0.583203125 retained 9,243 correct, whereas 0.5830078125 lost one prediction; the proposed conservative quarter-point tests for a calibration gain while staying closer to the successful endpoint, since the intervening 0.58310546875 attempts yielded only timeouts.

<<<<<<< SEARCH
            power = 0.583203125
=======
            power = 0.583154296875
>>>>>>> REPLACE