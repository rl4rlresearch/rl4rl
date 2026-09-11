MECHANISM: Conservative flip-ensemble power-mean boundary refinement

HYPOTHESIS: A 0.58316650390625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.583203125 to 0.58316650390625.

EVIDENCE: Order 0.583172607421875 retained 9,243 correct with the best verified cross-entropy, while 0.5830078125 lost one prediction; this nearby lower order previously timed out without contradictory accuracy evidence.

<<<<<<< SEARCH
            power = 0.583203125
=======
            power = 0.58316650390625
>>>>>>> REPLACE