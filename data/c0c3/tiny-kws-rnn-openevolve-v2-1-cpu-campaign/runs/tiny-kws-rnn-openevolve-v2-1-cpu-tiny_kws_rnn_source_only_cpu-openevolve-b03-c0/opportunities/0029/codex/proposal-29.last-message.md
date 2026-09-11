MECHANISM: Incremental earliest-frame recurrent step pruning

HYPOTHESIS: Omitting frames 0 and 1 will retain at least 85% validation accuracy while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 3.23% versus the current design.

INTENDED_EDIT: Schedule frames 2–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.

EVIDENCE: Omitting frame 0 improved accuracy from 85.52% to 85.64% while reducing execution to 31 steps, indicating that the earliest context is expendable and motivating an incremental test of the adjacent frame.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(2, available_frames))
>>>>>>> REPLACE