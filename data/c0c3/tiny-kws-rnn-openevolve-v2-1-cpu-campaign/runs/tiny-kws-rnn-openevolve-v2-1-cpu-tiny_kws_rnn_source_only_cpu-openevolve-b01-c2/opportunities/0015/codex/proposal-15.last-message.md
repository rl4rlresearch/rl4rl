MECHANISM: Single-frame leading-context pruning

HYPOTHESIS: The 58-unit GRU can omit only the earliest frame while retaining at least 85% validation accuracy and reducing total inference MACs from 355,092,240 to approximately 344,031,060.

INTENDED_EDIT: Change the causal frame schedule from all 32 frames to frames 1–31, preserving model capacity and temporal summaries while executing 31 recurrent steps.

EVIDENCE: The current 58-unit full-resolution model achieved 85.77% accuracy; since aggressive thinning to 24 frames failed, removing just one likely low-information boundary frame is a conservative probe that reduces more MACs than narrowing from 58 to 57 units.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE