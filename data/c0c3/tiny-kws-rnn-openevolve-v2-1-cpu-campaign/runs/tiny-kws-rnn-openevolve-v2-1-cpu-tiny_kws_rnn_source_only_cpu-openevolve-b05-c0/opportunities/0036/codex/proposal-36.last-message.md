MECHANISM: Anchored-prefix interior-frame ablation

HYPOTHESIS: Processing frames 3, 5–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 23 to 22 recurrent steps and total inference MACs by approximately 4.3%.

INTENDED_EDIT: Remove frame 4 while preserving frame 3 as the initial recurrent anchor, the remaining dense command-bearing prefix, and both informative late samples.

EVIDENCE: Removing frame 2 improved accuracy to 85.64%, whereas subsequently removing frame 3 fell to 84.66%; retaining frame 3 and ablating frame 4 directly tests whether the failure was specific to losing the earliest temporal anchor rather than requiring every early frame.

<<<<<<< SEARCH
        if available_frames >= 8:
            full_window = list(range(2, available_frames - 2))
            return full_window[1:-6] + full_window[-3::2]
        return list(range(available_frames))
=======
        if available_frames >= 8:
            full_window = list(range(2, available_frames - 2))
            schedule = full_window[1:-6] + full_window[-3::2]
            if len(schedule) > 2:
                return schedule[:1] + schedule[2:]
            return schedule
        return list(range(available_frames))
>>>>>>> REPLACE