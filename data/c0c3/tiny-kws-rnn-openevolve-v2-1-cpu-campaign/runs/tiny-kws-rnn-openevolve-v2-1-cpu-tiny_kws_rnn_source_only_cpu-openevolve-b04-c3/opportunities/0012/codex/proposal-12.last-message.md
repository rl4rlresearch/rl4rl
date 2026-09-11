MECHANISM: Early-boundary single-frame thinning

HYPOTHESIS: A 127-unit GRU that drops the first frame from the qualified 28-frame grid will reach at least 85% accuracy while matching the lower MACs and 27-step count of the prior combined design.

INTENDED_EDIT: Retain the existing 127-unit recurrent model but remove the earliest scheduled frame instead of the previously tested penultimate frame.

EVIDENCE: The 127-unit/28-step and 128-unit/27-step reductions qualified independently; their combination missed qualification by only three validation examples when frame 30 was removed. Preserving that near-terminal frame while dropping likely redundant boundary context directly tests whether temporal allocation caused the miss.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE