MECHANISM: Successive trailing-frame trimming

HYPOTHESIS: Processing frames 3–29 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 28 to 27 recurrent steps and total inference MACs from 644,493,850 to approximately 621,508,405.

INTENDED_EDIT: Omit one additional trailing input frame while preserving the successful recurrent architecture, dual-timescale readout, classifier, and training procedure.

EVIDENCE: The unified GRU improved to 87.61% accuracy after the previous trailing-frame removal, leaving a 2.61-point margin above the requirement; that edit also reduced one full recurrent step without harming accuracy, directly motivating another one-step trim.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 1))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 2))
>>>>>>> REPLACE