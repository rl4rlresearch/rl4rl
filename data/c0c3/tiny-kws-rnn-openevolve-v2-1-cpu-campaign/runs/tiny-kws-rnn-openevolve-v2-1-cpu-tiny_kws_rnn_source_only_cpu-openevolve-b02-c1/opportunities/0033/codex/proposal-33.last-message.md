MECHANISM: Successive trailing-frame trimming

HYPOTHESIS: Processing frames 3–28 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 27 to 26 recurrent steps and total inference MACs from 621,508,405 to approximately 598,522,960.

INTENDED_EDIT: Omit one additional trailing input frame while preserving the recurrent architecture, dual-timescale readout, classifier, and training procedure.

EVIDENCE: The same unified GRU remained above 87% accuracy after two consecutive trailing-frame removals, and the current 27-step model retains a 2.12-point margin above the requirement, directly motivating another one-step reduction.

<<<<<<< SEARCH
        return list(range(3, available_frames - 2))
=======
        return list(range(3, available_frames - 3))
>>>>>>> REPLACE