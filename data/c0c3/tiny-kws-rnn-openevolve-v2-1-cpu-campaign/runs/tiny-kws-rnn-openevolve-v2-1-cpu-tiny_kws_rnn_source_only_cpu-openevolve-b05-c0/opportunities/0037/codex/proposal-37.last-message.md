MECHANISM: Anchored-prefix interior-frame ablation

HYPOTHESIS: Processing frames 3, 6–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs by approximately 4.5%.

INTENDED_EDIT: Remove frame 5 while retaining frame 3 as the initial recurrent anchor, the remaining dense command-bearing region, and both informative late samples.

EVIDENCE: Removing frame 3 failed at 84.66%, but retaining frame 3 and removing frame 4 achieved 85.03%; this motivates testing whether the next early interior frame is also redundant without sacrificing the demonstrated anchor or late temporal coverage.

<<<<<<< SEARCH
                return schedule[:1] + schedule[2:]
=======
                return schedule[:1] + schedule[3:]
>>>>>>> REPLACE