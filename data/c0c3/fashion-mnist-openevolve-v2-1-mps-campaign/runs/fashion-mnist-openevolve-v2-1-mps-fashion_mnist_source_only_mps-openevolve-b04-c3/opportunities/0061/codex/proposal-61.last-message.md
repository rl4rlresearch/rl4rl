MECHANISM: Fine-grained sub-arithmetic probability power-mean flip ensemble

HYPOTHESIS: A 0.5875-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below 0.209057.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.60 to 0.5875.

EVIDENCE: Lowering the order from 0.625 to 0.60 retained 9,243 correct predictions and reduced cross-entropy from 0.2090745 to 0.2090570; the prior 0.5875 verification timed out and therefore provided no contradictory accuracy evidence.

<<<<<<< SEARCH
            power = 0.60
=======
            power = 0.5875
>>>>>>> REPLACE