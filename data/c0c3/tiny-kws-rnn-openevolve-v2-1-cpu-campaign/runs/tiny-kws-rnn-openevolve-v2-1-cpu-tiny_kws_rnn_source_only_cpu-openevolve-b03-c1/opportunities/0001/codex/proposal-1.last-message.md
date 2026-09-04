MECHANISM: Two-to-one causal frame decimation

HYPOTHESIS: Training and inference on 16 uniformly spaced causal frames will retain at least 85% validation accuracy because adjacent log-mel frames are highly redundant, while halving recurrent MACs and executed steps from the 87.24%-accurate baseline.

INTENDED_EDIT: Process every second input frame, selecting the odd-indexed frames so the schedule spans through the final frame.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per clip, leaving a 2.24-point accuracy margin for temporal downsampling.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames, 2))
>>>>>>> REPLACE