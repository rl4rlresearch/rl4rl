MECHANISM: Second leading-frame omission

HYPOTHESIS: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 28.3 million.

INTENDED_EDIT: Omit the first two input frames while preserving the final 30 causal frames, 18-feature mel compression, 99-unit recurrent state, classifier, and training procedure.

EVIDENCE: The current 31-step model achieved 85.28% accuracy, and omitting the first frame previously retained 85.03%. The prior 30-step attempt timed out without accuracy evidence, so removing one additional low-information boundary frame remains the most informative temporal-cost probe.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
>>>>>>> REPLACE