MECHANISM: Incremental trailing-boundary pruning

HYPOTHESIS: A 60-unit GRU processing frames 3–29 will retain at least 85% validation accuracy while reducing total inference MACs from 329,781,600 to approximately 318,045,600.

INTENDED_EDIT: Widen the current GRU from 58 to 60 units and adopt a 27-step causal schedule that removes one additional trailing boundary frame from the qualified 60-unit design.

EVIDENCE: The 60-unit model processing frames 3–30 achieved 86.50% accuracy at 329,781,600 MACs, while excluding frame 3 was worse; its 1.50-point margin motivates preserving the informative leading boundary and pruning the next trailing frame.

<<<<<<< SEARCH
        self.hidden_size = 58
=======
        self.hidden_size = 60
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 2))
>>>>>>> REPLACE