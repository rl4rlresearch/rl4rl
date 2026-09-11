MECHANISM: Nested three-of-four causal frame sampling

HYPOTHESIS: Processing 24 uniformly distributed frames with the unchanged 128-unit GRU will recover at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by 25% versus the qualified 32-frame design.

INTENDED_EDIT: Keep three frames from every four, including the first and final frames, using a schedule that is a strict superset of the unsuccessful 16-frame odd-index schedule.

EVIDENCE: The 32-frame model achieved 87.24% accuracy, while uniform 16-frame decimation achieved 82.58%; testing the 24-frame midpoint directly brackets the temporal-compute threshold while restoring eight evenly distributed observations.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [i for i in range(available_frames) if i % 4 != 2]
>>>>>>> REPLACE