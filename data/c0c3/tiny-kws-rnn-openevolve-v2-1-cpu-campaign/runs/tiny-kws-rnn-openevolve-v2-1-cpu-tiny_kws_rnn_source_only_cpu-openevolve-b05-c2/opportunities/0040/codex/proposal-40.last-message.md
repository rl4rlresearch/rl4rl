MECHANISM: Incremental boundary-frame temporal pruning

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 22 causal frames while reducing total inference MACs from 303,988,480 to approximately 290,844,160.

INTENDED_EDIT: Apply the qualified 23-step schedule and additionally omit the earliest interior frame, preserving both endpoints and reducing 32-frame execution to 22 recurrent steps.

EVIDENCE: The 23-step design achieved 86.26% accuracy with a 1.26-point margin after every adjacent reduction from 32 through 23 steps qualified; pruning one likely low-information boundary frame is the smallest next structural reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 4:
            skipped = {
                available_frames // 4,
                available_frames // 2,
                3 * available_frames // 4,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE