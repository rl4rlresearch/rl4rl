MECHANISM: Passing-schedule frame ablation

HYPOTHESIS: Removing only the earliest frame from the verified 28-frame schedule will retain at least 85% accuracy while reducing recurrent steps and MACs by approximately 3.6%.

INTENDED_EDIT: Preserve 27 of the exact frame indices used by the passing design, dropping frame 0 instead of recomputing a phase-shifted uniform 27-frame schedule.

EVIDENCE: The uniform 28-frame schedule passed at 85.276%, whereas the uniform 27-frame schedule failed at 84.294%; because recomputing 27 uniform indices changes many sampled locations, ablating one likely low-information leading frame from the passing schedule is a more controlled cost reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]
        return schedule[1:] if len(schedule) > 2 else schedule
>>>>>>> REPLACE