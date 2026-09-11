MECHANISM: Symmetric early-edge temporal pruning

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 23 causal frames while reducing total inference MACs from 317,132,800 to approximately 303,988,480.

INTENDED_EDIT: Apply the qualified 24-step schedule and additionally omit the frame at one-sixteenth of the sequence, symmetrically complementing the proven fifteen-sixteenths omission while retaining both endpoints.

EVIDENCE: The 24-step design achieved 86.26% accuracy with a 1.26-point margin after every adjacent reduction from 32 through 24 steps qualified; removing one edge-adjacent frame is the smallest and most informative next compute reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
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
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE