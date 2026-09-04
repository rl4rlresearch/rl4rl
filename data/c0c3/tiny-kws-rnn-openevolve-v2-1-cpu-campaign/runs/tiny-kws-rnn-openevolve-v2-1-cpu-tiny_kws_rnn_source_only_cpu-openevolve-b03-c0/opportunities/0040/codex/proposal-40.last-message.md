MECHANISM: Alternate-phase late-tail temporal thinning

HYPOTHESIS: Retaining frame 25 while omitting frames 26–27 will achieve at least 85% validation accuracy with 24 recurrent steps because the failed 24-step schedule omitted frame 25 while retaining frame 26.

INTENDED_EDIT: Change the 32-frame schedule from frames 3–26 and 28 to frames 3–25 and 28, reducing execution from 25 to 24 steps without changing model capacity or training.

EVIDENCE: The current frames 3–26 and 28 schedule passes at 85.03%, while frames 3–24, 26, and 28 scored 84.66%; swapping which adjacent late frame is retained directly tests whether frame 25 carries the information lost by that unsuccessful pruning.

<<<<<<< SEARCH
        schedule = list(range(3, available_frames - 3))
        del schedule[-2]
        return schedule
=======
        schedule = list(range(3, available_frames - 3))
        del schedule[-3:-1]
        return schedule
>>>>>>> REPLACE