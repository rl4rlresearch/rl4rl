MECHANISM: Evenly distributed seven-of-eight temporal sampling

HYPOTHESIS: Processing 28 evenly spaced frames will retain at least 85% validation accuracy while reducing recurrent MACs by exactly 12.5% versus the qualified 32-step baseline.

INTENDED_EDIT: Skip four interior frames distributed across the recording while preserving the first and final frames.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while the more aggressive 24-step schedule achieved 84.66%; testing the midpoint directly targets the cost–accuracy boundary with substantially more margin than the failed design.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames == 32:
            skipped = {6, 13, 20, 27}
            return [index for index in range(available_frames) if index not in skipped]
        return list(range(available_frames))
>>>>>>> REPLACE