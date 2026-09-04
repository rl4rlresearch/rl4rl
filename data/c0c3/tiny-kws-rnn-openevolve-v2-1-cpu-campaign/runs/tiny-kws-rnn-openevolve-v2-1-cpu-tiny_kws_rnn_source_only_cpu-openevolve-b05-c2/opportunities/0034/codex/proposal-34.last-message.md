MECHANISM: Incremental early-frame temporal pruning with signed-extrema readout

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 28 causal frames while reducing total inference MACs from 382,854,400 to approximately 369,710,080.

INTENDED_EDIT: Preserve the proven quarter, center, and three-quarter omissions and add one omission at one-eighth of the sequence, reducing 32-frame execution from 29 to 28 recurrent steps.

EVIDENCE: The 29-step design achieved 86.63% accuracy, matching the 30-step result and retaining a 1.63-point margin; this supports the smallest next temporal reduction while preserving the three already-qualified omissions.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 2:
            del schedule[available_frames // 2]
        return schedule
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                available_frames // 2,
                3 * available_frames // 4,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE