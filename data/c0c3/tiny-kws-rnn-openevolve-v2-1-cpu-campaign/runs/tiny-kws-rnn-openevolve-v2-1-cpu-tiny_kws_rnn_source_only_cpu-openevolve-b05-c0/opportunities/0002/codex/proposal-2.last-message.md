MECHANISM: Uniform 4:3 causal frame subsampling

HYPOTHESIS: Processing 24 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 25% versus the 32-step baseline.

INTENDED_EDIT: Replace the full 32-frame schedule with 24 monotonically increasing indices distributed across the complete recording, including the first and final frames.

EVIDENCE: Sixteen-frame 2× striding reduced accuracy from 87.24% to 83.07%, narrowly missing the threshold; retaining half of the discarded temporal evidence is the most informative intermediate cost-accuracy test.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = max(2, (3 * available_frames) // 4)
        return [
            ((step + 1) * available_frames) // steps - 1
            for step in range(steps)
        ]
>>>>>>> REPLACE