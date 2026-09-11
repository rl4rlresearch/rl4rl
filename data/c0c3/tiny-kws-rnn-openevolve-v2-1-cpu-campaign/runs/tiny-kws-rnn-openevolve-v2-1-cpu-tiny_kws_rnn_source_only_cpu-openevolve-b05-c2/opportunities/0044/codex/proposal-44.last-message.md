MECHANISM: Iterative dense-cluster temporal pruning

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy after additionally omitting frame 10, reducing execution from 21 to 20 recurrent steps and total inference MACs to approximately 264,555,520.

INTENDED_EDIT: Apply the qualified 21-step schedule and omit frame 10, an interior frame surrounded by retained frames 9 and 11, while preserving frame 29 and both endpoints.

EVIDENCE: The 21-step schedule qualified at 85.40% after frame 6 was removed from a dense retained cluster, whereas removing frame 29 failed at 84.91%; this motivates testing the same local pruning strategy at another densely sampled interior location.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(5 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE