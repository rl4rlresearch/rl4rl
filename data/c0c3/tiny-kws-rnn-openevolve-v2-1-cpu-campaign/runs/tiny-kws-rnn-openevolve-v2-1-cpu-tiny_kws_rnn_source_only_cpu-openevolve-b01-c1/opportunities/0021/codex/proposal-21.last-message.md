MECHANISM: Trailing-boundary frame pruning

HYPOTHESIS: Processing frames 4–30 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.6% versus the 28-step design.

INTENDED_EDIT: Preserve the earliest frame retained by the successful 28-step model while omitting the final frame, producing a 27-step causal schedule.

EVIDENCE: Frames 4–31 achieved 85.40% accuracy, whereas frames 5–31 fell to 83.68%; this suggests frame 4 may be load-bearing and makes removing the opposite boundary frame the most informative 27-step test.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 1))
>>>>>>> REPLACE