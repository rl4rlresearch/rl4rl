MECHANISM: Single-edge-frame causal trimming

HYPOTHESIS: Omitting only the first frame will retain at least 85% accuracy from the current 87.12% baseline while reducing recurrent MACs and executed steps by 3.125%.

INTENDED_EDIT: Process the most recent 31 of 32 frames, preserving uniform spacing and the final input frame.

EVIDENCE: Reductions to 24 and 16 frames were too aggressive, but the qualified 98-unit dual-readout model has 2.12 percentage points of accuracy margin; a one-frame edge reduction is the most conservative untested temporal optimization.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 31, 0)
        return list(range(start, available_frames))
>>>>>>> REPLACE