MECHANISM: Progressive early-interior frame ablation

HYPOTHESIS: Processing frames 3, 5, 6, 9–23, 27, and 29 will retain at least 85% validation accuracy while reducing recurrent execution from 21 to 20 steps.

INTENDED_EDIT: Remove frame 8 from the verified 21-step schedule while preserving the established early anchors, dense later region, and both late samples.

EVIDENCE: Removing frame 7 from the 22-step schedule achieved 85.15%, whereas removing frames 5 or 6 failed; frame 8 is therefore the next untested ablation that preserves all early frames shown to be important.

<<<<<<< SEARCH
                return schedule[:1] + schedule[2:4] + schedule[5:]
=======
                return schedule[:1] + schedule[2:4] + schedule[6:]
>>>>>>> REPLACE