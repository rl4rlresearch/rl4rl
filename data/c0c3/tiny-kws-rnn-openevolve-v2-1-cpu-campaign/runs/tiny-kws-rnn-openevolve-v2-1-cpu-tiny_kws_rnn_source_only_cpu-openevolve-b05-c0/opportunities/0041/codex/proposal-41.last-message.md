MECHANISM: Progressive early-interior frame ablation

HYPOTHESIS: Processing frames 3, 5, 6, 8, 10–23, 27, and 29 will retain at least 85% validation accuracy while reducing recurrent execution from 21 to 20 steps.

INTENDED_EDIT: Preserve frame 8 and remove frame 9 from the verified 21-step schedule.

EVIDENCE: Removing frame 7 achieved 85.15%, while removing frame 8 fell to 84.05%; ablating frame 9 next tests whether frame 8 is specifically important rather than indicating a general 20-step limit.

<<<<<<< SEARCH
            if len(schedule) > 2:
                return schedule[:1] + schedule[2:4] + schedule[5:]
=======
            if len(schedule) > 2:
                return schedule[:1] + schedule[2:4] + schedule[5:6] + schedule[7:]
>>>>>>> REPLACE