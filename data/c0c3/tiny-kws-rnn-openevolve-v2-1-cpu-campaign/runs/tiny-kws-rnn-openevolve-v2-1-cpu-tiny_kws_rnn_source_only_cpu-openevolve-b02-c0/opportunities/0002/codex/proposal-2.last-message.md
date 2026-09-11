MECHANISM: Three-of-four causal frame subsampling

HYPOTHESIS: Processing 24 uniformly distributed frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 25% versus the 32-frame baseline.

INTENDED_EDIT: Skip every fourth input frame while retaining the final frame, yielding 24 recurrent steps for the fixed 32-frame input.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while aggressive 16-step decimation achieved 82.58%; testing the 24-step midpoint is the most informative next reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return [
            index
            for index in range(available_frames)
            if index % 4 != 0 or index == available_frames - 1
        ]
>>>>>>> REPLACE