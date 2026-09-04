MECHANISM: Uniform core-frame subsampling with full-utterance coverage

HYPOTHESIS: Processing 26 evenly distributed frames across the 30-frame interior will retain at least 85% validation accuracy while reducing total inference MACs from 592,816,330 to approximately 571,090,060.

INTENDED_EDIT: Replace contiguous frames 1–27 with 26 uniformly spaced indices spanning frames 1 through 30, preserving seven causal readout bins while eliminating one recurrent step.

EVIDENCE: The current 27-step model achieved 85.77% accuracy, but progressive pruning discarded the utterance tail; distributing one fewer step across the full interior tests whether temporal coverage can preserve accuracy more efficiently than further trailing-frame removal.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(1, available_frames - 4))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 7:
            return list(range(available_frames))
        steps = available_frames - 6
        span = available_frames - 3
        return [
            1 + (index * span + (steps - 2) // 2) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE