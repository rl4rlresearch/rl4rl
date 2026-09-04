MECHANISM: Binary-search calibration of the flip-ensemble power mean

HYPOTHESIS: A 0.5828125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090463.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.584375 to 0.5828125.

EVIDENCE: Order 0.584375 achieved 9,243 correct with 0.2090463 cross-entropy, while 0.58125 achieved 9,242 correct with lower 0.2090442 cross-entropy; their midpoint is the most informative next probe of the accuracy boundary.

<<<<<<< SEARCH
            power = 0.584375
=======
            power = 0.5828125
>>>>>>> REPLACE