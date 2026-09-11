MECHANISM: Moderate endpoint-preserving temporal subsampling

HYPOTHESIS: Increasing the uniform schedule from 16 to 20 frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 37.5% versus the 32-frame baseline.

INTENDED_EDIT: Process 20 uniformly distributed causal frames spanning the complete recording.

EVIDENCE: The 16-frame schedule narrowly missed the target at 84.54%, only 0.46 percentage points short; four additional distributed observations should restore sufficient temporal detail while preserving most of the computational savings.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]
>>>>>>> REPLACE