MECHANISM: Uniform causal frame subsampling

HYPOTHESIS: Training and inference on 24 uniformly distributed frames will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 25%.

INTENDED_EDIT: Replace the full 32-frame schedule with a 24-step schedule spanning the complete recording, including the first and final frames.

EVIDENCE: The current 128-unit GRU achieves 87.24% accuracy but executes all 32 frames, with recurrence accounting for 1,482,178,560 of 1,483,013,120 MACs; removing eight temporally redundant steps directly targets the dominant cost with a 2.24-point accuracy margin.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(24, available_frames)
        if target_steps <= 1:
            return list(range(available_frames))
        return [
            i * (available_frames - 1) // (target_steps - 1)
            for i in range(target_steps)
        ]
>>>>>>> REPLACE