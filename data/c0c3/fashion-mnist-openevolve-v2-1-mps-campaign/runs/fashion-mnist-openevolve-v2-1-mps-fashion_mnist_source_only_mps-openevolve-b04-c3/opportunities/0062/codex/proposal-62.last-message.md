MECHANISM: 0.575-order probability power-mean flip ensemble

HYPOTHESIS: A 0.575-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below the verified 0.5875-order result of 0.2090484.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.625 to 0.575.

EVIDENCE: Orders 0.625, 0.60, and 0.5875 all achieved 9,243 correct while progressively lowering cross-entropy; 0.575 continues this trend by a conservative 0.0125 step.

<<<<<<< SEARCH
            power = 0.625
=======
            power = 0.575
>>>>>>> REPLACE