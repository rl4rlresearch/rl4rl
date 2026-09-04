MECHANISM: Successive trailing-frame trimming

HYPOTHESIS: Processing frames 3–27 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and total inference MACs from 598,522,960 to approximately 575,537,515.

INTENDED_EDIT: Omit one additional trailing input frame while preserving the recurrent architecture, dual-timescale readout, classifier, and training procedure.

EVIDENCE: Three consecutive trailing-frame removals retained at least 86.87% accuracy; the current 26-step model remains 1.87 points above the requirement, directly motivating another one-step reduction.

<<<<<<< SEARCH
        return list(range(3, available_frames - 3))
=======
        return list(range(3, available_frames - 4))
>>>>>>> REPLACE