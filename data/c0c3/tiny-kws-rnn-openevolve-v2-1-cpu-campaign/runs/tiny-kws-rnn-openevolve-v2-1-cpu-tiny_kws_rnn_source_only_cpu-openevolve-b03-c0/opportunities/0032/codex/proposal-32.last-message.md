MECHANISM: Opposite-end frame pruning

HYPOTHESIS: Omitting frames 0–2 and the final frame will retain at least 85% validation accuracy while matching the failed 28-step model’s lower inference cost, because it restores frame 3 and removes a likely trailing-context frame instead.

INTENDED_EDIT: Run the verified recurrent model on frames 3 through 30 for standard 32-frame inputs, preserving 28 causal recurrent steps.

EVIDENCE: The 29-step schedule omitting frames 0–2 passed at 85.52%, while additionally omitting frame 3 narrowly failed at 84.79%; relocating the fourth omission to the opposite temporal boundary directly tests whether frame 3 is more informative than the final frame.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(3, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(3, available_frames - 1))
>>>>>>> REPLACE