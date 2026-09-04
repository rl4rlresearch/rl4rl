MECHANISM: Conservative flip-ensemble power-mean refinement

HYPOTHESIS: A 0.58316802978515625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.5831695556640625 to 0.58316802978515625.

EVIDENCE: The current order retains 9,243 correct with the best verified cross-entropy, while the nearby lower order 0.58316650390625 timed out without contradictory accuracy evidence; their midpoint is the most conservative informative refinement.

<<<<<<< SEARCH
            power = 0.5831695556640625
=======
            power = 0.58316802978515625
>>>>>>> REPLACE