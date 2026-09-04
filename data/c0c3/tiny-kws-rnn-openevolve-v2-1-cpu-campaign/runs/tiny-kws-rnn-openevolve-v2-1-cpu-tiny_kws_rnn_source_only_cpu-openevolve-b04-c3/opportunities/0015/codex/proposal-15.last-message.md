MECHANISM: Sharpened-loss early-boundary frame thinning

HYPOTHESIS: A 127-unit GRU dropping the earliest frame and reducing label smoothing from 0.03 to 0.01 will recover the single validation example needed for at least 85% accuracy while lowering inference to approximately 1.233B MACs and 27 steps.

INTENDED_EDIT: Remove frame 0 from the qualified 28-frame schedule and use less-smoothed cross-entropy to strengthen narrow decision margins without changing inference cost.

EVIDENCE: The prior 127-unit variant dropping frame 0 achieved 84.91%, only one validation example below qualification and better than the other tested 27-frame schedules; the qualified 127-unit/28-step model shows sufficient capacity, motivating a training-only margin adjustment at the lower-cost boundary.

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

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    return F.cross_entropy(logits, labels, label_smoothing=0.01)
>>>>>>> REPLACE