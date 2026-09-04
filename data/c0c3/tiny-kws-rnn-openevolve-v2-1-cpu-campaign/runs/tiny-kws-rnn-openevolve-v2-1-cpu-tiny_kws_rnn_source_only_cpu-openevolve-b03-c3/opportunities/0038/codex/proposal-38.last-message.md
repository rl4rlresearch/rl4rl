MECHANISM: Balanced seven-bin uniform core subsampling

HYPOTHESIS: Processing 25 uniformly distributed frames across frames 1–30 while balancing their readouts across seven temporal bins will retain at least 85% validation accuracy and reduce total inference MACs from 571,090,060 to approximately 549,363,790.

INTENDED_EDIT: Reduce the qualified uniform schedule from 26 to 25 recurrent steps and replace four-step bin assignment—which would create a singleton final bin—with near-uniform seven-bin assignment.

EVIDENCE: The 26-step uniformly distributed model achieved 85.40% accuracy; the 27- and 28-step models also qualified, while their final temporal bins contained three and four observations respectively. Rebalancing 25 observations avoids an under-supported singleton readout while testing the next step reduction.

<<<<<<< SEARCH
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=6)
=======
        bin_index = (count.to(dtype=torch.long) * 7 // 25).clamp(max=6)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_indices = (positions // 4).clamp(max=6)
=======
        bin_indices = (positions * 7 // 25).clamp(max=6)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(1, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        steps = available_frames - 7
        span = available_frames - 3
        return [
            1 + (index * span + (steps - 2) // 2) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE