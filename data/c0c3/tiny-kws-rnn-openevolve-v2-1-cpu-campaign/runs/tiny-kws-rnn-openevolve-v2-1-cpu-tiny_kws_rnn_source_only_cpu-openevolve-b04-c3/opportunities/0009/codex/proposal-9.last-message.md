MECHANISM: Combined one-unit width and one-frame thinning

HYPOTHESIS: A 127-unit GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing exact recurrent MACs below the qualified 128-unit, 27-step design.

INTENDED_EDIT: Remove the penultimate frame from the current 28-frame uniform schedule, preserving both endpoints and the existing 127-unit recurrent state.

EVIDENCE: The reductions qualify independently: 127 units at 28 steps achieved 85.28% accuracy, and 128 units at 27 steps achieved 85.15%; combining them directly tests the next structural cost boundary.

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
            schedule.pop(-2)
        return schedule
>>>>>>> REPLACE