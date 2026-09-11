MECHANISM: Span-preserving temporal thinning with causal deltas

HYPOTHESIS: The qualified 74-unit delta-augmented GRU will retain at least 85% validation accuracy using 27 evenly distributed frames across the latest 28-frame window, while reducing recurrent steps from 28 to 27 and total inference MACs to approximately 557.9 million.

INTENDED_EDIT: Replace the contiguous 28-frame schedule with 27 causally ordered samples spanning the same latest 28-frame window.

EVIDENCE: The current 74-unit delta model achieved 86.01% accuracy with 28 frames, leaving 1.01 points of margin; separately, span-preserving thinning allowed the 88-unit model to qualify at 26 steps and produced better cross-entropy than its contiguous 27-frame schedule.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 28, 0)
        return list(range(start, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
        ]
>>>>>>> REPLACE