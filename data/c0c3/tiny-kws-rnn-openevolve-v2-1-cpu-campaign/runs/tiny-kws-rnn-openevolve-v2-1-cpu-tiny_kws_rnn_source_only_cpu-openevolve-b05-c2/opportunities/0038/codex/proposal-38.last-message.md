MECHANISM: Incremental late-edge temporal pruning

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 24 causal frames while reducing total inference MACs from 330,277,120 to approximately 317,132,800.

INTENDED_EDIT: Restore the proven seven uniformly spaced omissions and additionally omit the frame at fifteen-sixteenths of the sequence, retaining both endpoints and reducing execution to 24 steps.

EVIDENCE: The 25-step design achieved 86.13% accuracy, and every incremental reduction from 32 through 25 steps qualified; its 1.13-point margin motivates testing the adjacent step boundary with a low-information edge-adjacent omission.

<<<<<<< SEARCH
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
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE