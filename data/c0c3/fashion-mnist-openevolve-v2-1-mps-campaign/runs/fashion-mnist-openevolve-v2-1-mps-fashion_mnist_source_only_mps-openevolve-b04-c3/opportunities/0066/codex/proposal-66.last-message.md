MECHANISM: Binary-search calibration of the flip-ensemble power mean

HYPOTHESIS: A 0.58359375-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090463.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.5875 to 0.58359375.

EVIDENCE: Order 0.584375 achieved 9,243 correct at 0.2090463 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.2090452; their midpoint is the most informative probe of the accuracy boundary.

<<<<<<< SEARCH
            power = 0.5875
=======
            power = 0.58359375
>>>>>>> REPLACE