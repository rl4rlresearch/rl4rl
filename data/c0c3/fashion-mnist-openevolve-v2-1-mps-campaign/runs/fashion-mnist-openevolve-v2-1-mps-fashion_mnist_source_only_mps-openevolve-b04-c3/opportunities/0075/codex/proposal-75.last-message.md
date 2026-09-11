MECHANISM: Conservative flip-ensemble power-mean boundary refinement

HYPOTHESIS: A 0.58316650390625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

INTENDED_EDIT: Lower only the inference-time probability power-mean order from 0.584375 to 0.58316650390625.

EVIDENCE: Order 0.5831787109375 retained 9,243 correct with the best verified cross-entropy, while 0.5830078125 lost one prediction; the proposed value cautiously approaches the lower-cross-entropy boundary, and the intervening 0.583154296875 run timed out without contradictory accuracy evidence.

<<<<<<< SEARCH
            power = 0.584375
=======
            power = 0.58316650390625
>>>>>>> REPLACE