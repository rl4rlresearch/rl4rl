MECHANISM: Opposite-edge temporal pruning

HYPOTHESIS: Retaining frame 28 while dropping frame 1 will recover at least 85% validation accuracy with 27 recurrent steps and approximately 561,166,620 total inference MACs.

INTENDED_EDIT: Change the schedule from frames 1–28 to frames 2–28, preserving the qualified recurrent architecture, spectral selection, and training procedure.

EVIDENCE: The 28-step frames-1–28 model achieved 86.87% accuracy, while frames 1–27 narrowly missed at 84.91%; this tests whether losing the latest frame, rather than using only 27 steps, caused the regression.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))
>>>>>>> REPLACE