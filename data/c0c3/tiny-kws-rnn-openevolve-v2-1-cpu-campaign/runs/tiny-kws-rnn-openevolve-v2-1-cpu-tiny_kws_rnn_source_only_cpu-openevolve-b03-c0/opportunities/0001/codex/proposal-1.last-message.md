MECHANISM: Uniform causal frame subsampling

HYPOTHESIS: Processing 24 uniformly distributed frames instead of all 32 will reduce recurrent MACs and executed steps by 25% while retaining at least 85% validation accuracy because adjacent log-mel frames are temporally redundant.

INTENDED_EDIT: Replace the full-frame schedule with 24 unique, increasing indices spanning the complete recording, including the first and final frames.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving a 2.24-point margin for reducing temporal computation.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(24, available_frames)
        if target_steps < 2:
            return list(range(available_frames))
        return [
            step * (available_frames - 1) // (target_steps - 1)
            for step in range(target_steps)
        ]
>>>>>>> REPLACE