MECHANISM: Progressive terminal-boundary trimming

HYPOTHESIS: Processing frames 2–29 with the verified 80-unit GRU will achieve at least 85% validation accuracy while reducing recurrent execution and MACs from 29 to 28 steps.

INTENDED_EDIT: Omit one additional terminal frame while preserving the successful model capacity, leading-frame offset, temporal-mean readout, and training procedure.

EVIDENCE: Frames 2–30 achieved 85.52% accuracy at 29 steps, outperforming the 30-step frames 2–31 schedule’s 85.28%; this indicates the terminal boundary is the most promising location for the next single-frame reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 1))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 2))
>>>>>>> REPLACE