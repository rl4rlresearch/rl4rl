MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.

INTENDED_EDIT: Change the explicit causal schedule from frames 1–31 to frames 2–31, yielding 30 recurrent steps while preserving model capacity and training.

EVIDENCE: The 31-step model achieved 85.40% accuracy; previous intended 30-step tests either timed out or reproduced the unchanged 31-step implementation, so this remains the smallest unresolved temporal-compression test.

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