MECHANISM: Binary-search calibration of the flip-ensemble power mean

HYPOTHESIS: A 0.583203125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.209045771.

INTENDED_EDIT: Raise only the inference-time probability power-mean order from 0.58125 to 0.583203125, leaving the verified architecture and training procedure unchanged.

EVIDENCE: Order 0.58359375 achieved 9,243 correct at 0.209045771 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.209045237; their midpoint remains the most informative boundary probe because its prior verification timed out without accuracy evidence.

<<<<<<< SEARCH
            power = 0.58125
=======
            power = 0.583203125
>>>>>>> REPLACE