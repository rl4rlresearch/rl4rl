MECHANISM: Incremental trailing-frame recurrent step pruning

HYPOTHESIS: Omitting frame 30 in addition to frames 0–2 and 31 will retain at least 85% validation accuracy while reducing recurrent execution from 28 to 27 steps and recurrent MACs by approximately 3.57%.

INTENDED_EDIT: Schedule frames 3–29 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.

EVIDENCE: The 29-step schedule using frames 3–31 passed at 85.52%; removing the final frame also preserved 85.52%, whereas removing frame 3 failed at 84.79%. This indicates trailing context is the better-supported target for the next step reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(3, available_frames - 1))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(3, available_frames - 2))
>>>>>>> REPLACE