MECHANISM: Incremental leading-context pruning at the qualified capacity boundary

HYPOTHESIS: The 60-unit GRU can omit frame 3 and process frames 4–31 while retaining at least 85% validation accuracy, reducing total dense inference MACs from 341,517,600 to approximately 329,781,600.

INTENDED_EDIT: Narrow the current GRU from 68 to 60 units and use the 28-step causal schedule covering frames 4–31, while preserving the proven mean, maximum, and final-state summaries and training procedure.

EVIDENCE: The qualified 60-unit model processing frames 3–31 achieved 85.40% accuracy at 341,517,600 MACs. Removing one additional leading boundary frame is the smallest direct probe beyond that result and saves one complete recurrent step per example.

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
        return list(range(4, available_frames))
>>>>>>> REPLACE