MECHANISM: Progressive early-interior frame ablation

HYPOTHESIS: Processing frames 3, 5, 6, 8, 9, 11–23, 27, and 29 will retain at least 85% validation accuracy while reducing recurrent execution from 21 to 20 steps.

INTENDED_EDIT: Preserve frames 8 and 9 while removing frame 10 from the verified 21-step schedule.

EVIDENCE: Removing frame 7 achieved 85.15%, while removing frame 8 scored 84.05% and removing frame 9 scored 84.42%; testing frame 10 is the next direct ablation that preserves every early frame currently implicated as important.

<<<<<<< SEARCH
                return schedule[:1] + schedule[2:4] + schedule[5:]
=======
                return schedule[:1] + schedule[2:4] + schedule[5:7] + schedule[8:]
>>>>>>> REPLACE