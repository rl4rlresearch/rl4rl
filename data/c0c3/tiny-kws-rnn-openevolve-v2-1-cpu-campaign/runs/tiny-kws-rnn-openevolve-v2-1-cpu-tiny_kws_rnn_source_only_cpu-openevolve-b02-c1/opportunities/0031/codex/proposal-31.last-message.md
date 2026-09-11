MECHANISM: Opposite-boundary frame trimming on the delta-augmented unified GRU

HYPOTHESIS: Processing frames 3–30 with the current 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 29 to 28 recurrent steps and total inference MACs to approximately 644,493,850.

INTENDED_EDIT: Omit the final input frame while preserving the successful recurrent architecture, readout, and training procedure.

EVIDENCE: The current unified GRU achieved 87.24% accuracy and substantially lower cross-entropy than the prior dual-GRU design. Removing the final frame from that weaker design reduced accuracy by only 0.74 points, so the current 2.24-point margin provides a grounded basis for retesting the larger step-level MAC reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 1))
>>>>>>> REPLACE