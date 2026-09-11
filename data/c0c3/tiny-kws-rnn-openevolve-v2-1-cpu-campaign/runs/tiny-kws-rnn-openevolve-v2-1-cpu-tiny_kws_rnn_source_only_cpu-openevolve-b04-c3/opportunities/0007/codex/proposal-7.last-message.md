MECHANISM: Near-terminal single-frame thinning

HYPOTHESIS: A 128-unit GRU using 27 of the qualified 28 scheduled frames will retain at least 85% validation accuracy while reducing MACs and recurrent steps below the qualified 127-unit, 28-step design.

INTENDED_EDIT: Use the qualified 28-frame uniform grid but remove its penultimate frame, preserving both endpoints and all other sampled frames.

EVIDENCE: The 128-unit model achieved 85.52% at 28 frames, whereas 24 frames achieved 83.19%; testing a minimal one-frame reduction isolates the remaining temporal-resolution boundary while retaining more accuracy margin than the 127-unit design.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
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