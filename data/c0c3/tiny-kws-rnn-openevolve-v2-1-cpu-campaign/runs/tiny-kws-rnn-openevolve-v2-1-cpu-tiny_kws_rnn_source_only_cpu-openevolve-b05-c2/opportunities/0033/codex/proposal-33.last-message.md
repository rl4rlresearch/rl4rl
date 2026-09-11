MECHANISM: Three-point temporal pruning with signed-extrema readout

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 29 causal frames while reducing total inference MACs from 395,998,720 to approximately 382,854,400.

INTENDED_EDIT: Add a third, evenly separated interior omission at three-quarters of the sequence, preserving both endpoints and reducing execution from 30 to 29 recurrent steps for 32-frame inputs.

EVIDENCE: The current 30-step model achieved 86.63% accuracy, 1.63 points above threshold, after the preceding 31-step model achieved 86.87%; this supports testing the smallest next temporal reduction with omissions distributed across the sequence.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 3:
            skipped = {available_frames // 4, available_frames // 2}
            schedule = [index for index in schedule if index not in skipped]
        return schedule
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 4:
            skipped = {
                available_frames // 4,
                available_frames // 2,
                3 * available_frames // 4,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE