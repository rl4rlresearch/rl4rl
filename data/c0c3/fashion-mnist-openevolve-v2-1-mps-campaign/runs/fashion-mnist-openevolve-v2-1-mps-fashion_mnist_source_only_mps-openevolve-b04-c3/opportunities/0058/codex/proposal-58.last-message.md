MECHANISM: 0.60-order probability power-mean flip ensemble

HYPOTHESIS: Lowering the power-mean order from 0.75 to 0.60 will retain at least 9,243 correct predictions while reducing cross-entropy below 0.2090745.

INTENDED_EDIT: Change only the inference-time probability power-mean order to 0.60, preserving the verified training procedure, architecture, schedule, and parameter count.

EVIDENCE: Order 0.625 achieved the best result with 9,243 correct and 0.2090745 cross-entropy, improving both metrics over order 0.75; order 0.60 is a conservative continuation toward the lower-cross-entropy geometric ensemble.

<<<<<<< SEARCH
            power = 0.75
=======
            power = 0.60
>>>>>>> REPLACE