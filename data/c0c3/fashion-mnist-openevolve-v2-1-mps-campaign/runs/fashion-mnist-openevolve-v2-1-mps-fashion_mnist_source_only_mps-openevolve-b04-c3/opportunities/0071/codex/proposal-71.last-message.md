MECHANISM: Binary-search calibration of the flip-ensemble power mean

HYPOTHESIS: A 0.58310546875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.584375 to 0.58310546875, leaving training unchanged.

EVIDENCE: Order 0.583203125 retained 9,243 correct at 0.2090455017 cross-entropy, while 0.5830078125 achieved lower cross-entropy but lost one correct prediction; their midpoint is the most informative boundary probe.

<<<<<<< SEARCH
            power = 0.584375
=======
            power = 0.58310546875
>>>>>>> REPLACE