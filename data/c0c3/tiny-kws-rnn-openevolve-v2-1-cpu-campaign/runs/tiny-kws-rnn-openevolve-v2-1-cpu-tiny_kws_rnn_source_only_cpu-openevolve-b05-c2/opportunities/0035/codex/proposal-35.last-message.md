MECHANISM: Symmetric late-frame temporal pruning

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 27 causal frames while reducing total inference MACs from 369,710,080 to approximately 356,565,760.

INTENDED_EDIT: Preserve the four proven omissions and additionally omit the frame at seven-eighths of the sequence, reducing 32-frame execution from 28 to 27 recurrent steps while retaining both endpoints.

EVIDENCE: The 28-step design achieved 86.87% accuracy, 1.87 points above threshold, after every incremental reduction from 32 through 28 steps qualified; adding the late counterpart to the proven one-eighth omission is the smallest next compute reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                available_frames // 2,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE