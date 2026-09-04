MECHANISM: Higher-update training for nested 28-step recurrence

HYPOTHESIS: Doubling the optimizer-update frequency with batch size 64 will recover the 0.215-point accuracy deficit of the nested 28-step design, reaching at least 85% while reducing recurrent MACs by approximately 3.45% versus the qualified 29-step model.

INTENDED_EDIT: Use batch size 64 and remove the second selected frame from the qualified 29-frame schedule, yielding the previously tested nested 28-step inference path.

EVIDENCE: The 111-unit nested 28-step model reached 84.785% versus the 85% requirement, with cross-entropy nearly identical to the qualified 29-step model; retaining the same 50,000-example exposure at half the batch size approximately doubles optimizer steps without adding inference MACs.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
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
        if len(schedule) > 2:
            del schedule[1]
        return schedule
>>>>>>> REPLACE