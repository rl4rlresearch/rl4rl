MECHANISM: Nested leading-edge frame omission

HYPOTHESIS: Removing frame 0 instead of frame 1 from the qualified 111-unit, 29-frame design will retain at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.45%.

INTENDED_EDIT: Use the qualified uniform 29-frame schedule as a base, then omit its first frame to execute 28 recurrent steps while preserving frame 1 and every qualified interior and final frame.

EVIDENCE: The qualified 29-step model achieved 85.03% accuracy; the nested 28-step variant that removed frame 1 narrowly missed at 84.785%, only two validation examples short, motivating an isolated test of the alternative leading-edge omission at identical inference cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(30, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        return schedule[1:] if len(schedule) > 2 else schedule
>>>>>>> REPLACE