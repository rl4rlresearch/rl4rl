MECHANISM: Binary-search calibration of the flip-ensemble power mean

HYPOTHESIS: A 0.5830078125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.583203125 to 0.5830078125.

EVIDENCE: Order 0.583203125 achieved 9,243 correct at 0.2090455017 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.2090452370; their midpoint is the most informative next probe of the accuracy boundary.

<<<<<<< SEARCH
            power = 0.583203125
=======
            power = 0.5830078125
>>>>>>> REPLACE