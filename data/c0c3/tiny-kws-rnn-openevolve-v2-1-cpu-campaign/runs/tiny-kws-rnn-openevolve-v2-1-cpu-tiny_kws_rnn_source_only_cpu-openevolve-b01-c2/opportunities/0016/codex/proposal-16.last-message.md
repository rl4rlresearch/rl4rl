MECHANISM: Capacity-for-time exchange via leading-context pruning

HYPOTHESIS: The stronger 60-unit GRU can process only frames 3–31 while retaining at least 85% validation accuracy and reducing total inference MACs from 344,031,060 to approximately 341,517,600.

INTENDED_EDIT: Preserve the qualified 60-unit recurrent model and temporal summaries, but omit the first three likely low-information boundary frames, reducing execution from 32 to 29 recurrent steps.

EVIDENCE: The 60-unit full-resolution model achieved 87.12% accuracy, while the narrower 58-unit model remained qualified after omitting the earliest frame at 85.03%; retaining two additional hidden units provides capacity to test two further leading-frame removals at lower MAC cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
>>>>>>> REPLACE