MECHANISM: Incremental earliest-frame recurrent step pruning

HYPOTHESIS: Omitting frames 0–3 will retain at least 85% validation accuracy while reducing recurrent execution from 29 to 28 steps and recurrent MACs by approximately 3.45%.

INTENDED_EDIT: Schedule frames 4–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.

EVIDENCE: Removing each of the first three frames successively preserved passing accuracy, with the latest 29-step model reaching 85.52%; this supports testing whether the next-earliest frame is also redundant.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(3, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))
>>>>>>> REPLACE