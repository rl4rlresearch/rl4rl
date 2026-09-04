MECHANISM: Uniform causal temporal subsampling

HYPOTHESIS: A 65-unit GRU processing 30 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 6.2% versus the verified 32-step model.

INTENDED_EDIT: Preserve the verified 65-unit recurrent capacity and training procedure, but uniformly select 30 frames spanning the complete causal one-second input.

EVIDENCE: The 65-unit, 32-step model achieved 86.50% accuracy, while both tested 64-unit variants failed; this motivates preserving recurrent width and probing modest temporal redundancy instead.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(30, available_frames)
        return [
            step * (available_frames - 1) // (target_steps - 1)
            for step in range(target_steps)
        ]
>>>>>>> REPLACE