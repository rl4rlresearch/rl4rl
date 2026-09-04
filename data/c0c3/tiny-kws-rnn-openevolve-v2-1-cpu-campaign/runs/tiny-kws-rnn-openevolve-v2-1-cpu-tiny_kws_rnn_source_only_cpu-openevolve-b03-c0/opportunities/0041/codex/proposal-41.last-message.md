MECHANISM: Anchor-preserving early-neighbor thinning

HYPOTHESIS: Retaining the important frame 3 while omitting adjacent frame 4 will preserve at least 85% validation accuracy with 24 recurrent steps.

INTENDED_EDIT: Change the standard 32-frame schedule from frames 3–26 and 28 to frames 3, 5–26, and 28.

EVIDENCE: Removing frame 3 caused a 28-step model to fail, while preserving informative frame 28 and removing its adjacent frame 27 produced the passing 25-step design; this tests the analogous adjacent-frame redundancy around the early anchor without sacrificing late coverage.

<<<<<<< SEARCH
        schedule = list(range(3, available_frames - 3))
        del schedule[-2]
        return schedule
=======
        schedule = list(range(3, available_frames - 3))
        del schedule[-2]
        if len(schedule) > 2:
            del schedule[1]
        return schedule
>>>>>>> REPLACE