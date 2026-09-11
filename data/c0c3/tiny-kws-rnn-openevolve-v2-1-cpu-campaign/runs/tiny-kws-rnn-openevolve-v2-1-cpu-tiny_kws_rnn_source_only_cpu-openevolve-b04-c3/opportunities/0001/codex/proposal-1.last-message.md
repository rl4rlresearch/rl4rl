MECHANISM: Uniform causal frame thinning

HYPOTHESIS: Processing 24 uniformly spaced frames instead of all 32 will preserve validation accuracy at or above 85% while reducing recurrent MACs and executed recurrent steps by 25%.

INTENDED_EDIT: Replace the full-frame schedule with 24 unique, uniformly distributed causal indices spanning the complete recording.

EVIDENCE: The 128-unit GRU reaches 87.24% accuracy using all 32 frames, providing a 2.24-point margin above the target; temporal redundancy is therefore the lowest-risk first target for structural cost reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE