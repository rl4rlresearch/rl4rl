MECHANISM: Symmetric boundary-frame pruning

HYPOTHESIS: Processing the central 30 contiguous frames will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs versus the verified 31-step model.

INTENDED_EDIT: Change the schedule from frames 1–31 to frames 1–30, removing the remaining outer boundary frame while preserving the 58-unit GRU and training procedure.

EVIDENCE: The 58-unit model passed at 85.15% when dropping the earliest boundary frame, whereas omitting an interior frame fell to 84.79%; this indicates boundary frames are less valuable and motivates pruning the opposite boundary next.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(31, available_frames)
        return list(range(available_frames - steps, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(30, available_frames)
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
>>>>>>> REPLACE