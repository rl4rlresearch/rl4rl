MECHANISM: Incremental trailing-boundary pruning

HYPOTHESIS: A 60-unit GRU processing frames 3–28 will retain at least 85% validation accuracy while reducing total inference MACs from 318,045,600 to approximately 306,309,600.

INTENDED_EDIT: Use the qualified 60-unit architecture and remove one additional trailing frame, reducing recurrent execution from 27 to 26 steps.

EVIDENCE: The 60-unit model processing frames 3–29 achieved 86.26% accuracy at 318,045,600 MACs, only 0.25 points below the 28-step design; this margin and stability motivate one more trailing-frame reduction.

<<<<<<< SEARCH
        self.hidden_size = 68
=======
        self.hidden_size = 60
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
>>>>>>> REPLACE