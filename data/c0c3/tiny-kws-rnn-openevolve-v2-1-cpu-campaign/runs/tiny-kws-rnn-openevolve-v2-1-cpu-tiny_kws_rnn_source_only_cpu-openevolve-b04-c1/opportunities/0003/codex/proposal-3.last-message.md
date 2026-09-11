MECHANISM: Conservative uniform causal frame decimation

HYPOTHESIS: Using 28 uniformly spaced frames will exceed 85% validation accuracy while reducing recurrent steps and MACs by 12.5% versus the verified 32-step baseline.

INTENDED_EDIT: Select 28 unique, evenly distributed causal frames spanning the complete recording.

EVIDENCE: The 26-step model narrowly missed the target at 84.66%, while the 32-step baseline reached 87.24%; 28 steps conservatively explores the boundary while retaining a meaningful compute reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE