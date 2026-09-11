MECHANISM: 0.5625-order probability power-mean flip ensemble

HYPOTHESIS: A 0.5625-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below 0.2090745.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.625 to 0.5625.

EVIDENCE: Reducing the order from 0.75 to 0.625 improved correct predictions from 9,242 to 9,243 and cross-entropy from 0.209166 to 0.209075; 0.5625 conservatively continues that trend while remaining above the unverified 0.5-order attempt.

<<<<<<< SEARCH
            power = 0.625
=======
            power = 0.5625
>>>>>>> REPLACE