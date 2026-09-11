MECHANISM: Interior-frame temporal thinning

HYPOTHESIS: Retaining frame 28 while omitting the locally redundant middle frame 16 will recover at least 85% accuracy at the lower 25-step inference cost.

INTENDED_EDIT: Change the 32-frame schedule from contiguous frames 3–28 to frames 3–15 and 17–28, leaving the model and training procedure unchanged.

EVIDENCE: The 26-step schedule over frames 3–28 reached 85.40%, while truncating it to frames 3–27 missed the threshold by only one validation example; preserving the later endpoint and relocating the omission to an interior frame tests whether temporal coverage matters more than one densely sampled middle frame.

<<<<<<< SEARCH
        return list(range(3, available_frames - 3))
=======
        scheduled = list(range(3, available_frames - 3))
        if len(scheduled) > 2:
            del scheduled[len(scheduled) // 2]
        return scheduled
>>>>>>> REPLACE