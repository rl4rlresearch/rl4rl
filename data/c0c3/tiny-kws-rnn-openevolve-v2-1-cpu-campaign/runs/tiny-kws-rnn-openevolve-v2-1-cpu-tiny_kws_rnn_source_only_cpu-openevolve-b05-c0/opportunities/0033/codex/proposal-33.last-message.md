MECHANISM: Earliest-prefix frame ablation

HYPOTHESIS: Processing frames 3–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 24 to 23 recurrent steps and total inference MACs by approximately 4.2%.

INTENDED_EDIT: Remove frame 2 from the verified 24-step schedule while preserving its dense command-bearing prefix and both informative late samples.

EVIDENCE: A 29-step schedule beginning at frame 3 achieved 85.28%, showing frame 2 is not intrinsically required, while removing frame 29 from the current schedule failed at 84.29%; this makes the earliest prefix frame a better-supported 23-step ablation than removing late coverage.

<<<<<<< SEARCH
            return full_window[:-6] + full_window[-3::2]
=======
            return full_window[1:-6] + full_window[-3::2]
>>>>>>> REPLACE