MECHANISM: Causal three-of-four temporal frame sampling

HYPOTHESIS: Reducing execution from 32 to 24 recurrent steps will cut recurrent MACs by exactly 25% while retaining at least 85% accuracy because the 0.8724 baseline provides margin and adjacent log-mel frames are temporally redundant.

INTENDED_EDIT: Skip one frame from each four-frame group while preserving the first and final frames.

EVIDENCE: The current 128-unit GRU reaches 0.872392638 validation accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, making temporal thinning the highest-leverage isolated reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [
            index
            for index in range(available_frames)
            if index % 4 != 1
        ]
>>>>>>> REPLACE