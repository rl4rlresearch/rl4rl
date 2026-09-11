MECHANISM: Iterative earliest-prefix frame ablation

HYPOTHESIS: Processing frames 4–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 23 to 22 recurrent steps and total inference MACs by approximately 4.3%.

INTENDED_EDIT: Remove frame 3 from the verified schedule while preserving the dense command-bearing region and both informative late samples.

EVIDENCE: Removing frame 2 from the verified 24-step schedule improved accuracy from 85.28% to 85.64%, while removing late frame 29 failed at 84.29%; this supports testing the next earliest frame without sacrificing late temporal coverage.

<<<<<<< SEARCH
            return full_window[1:-6] + full_window[-3::2]
=======
            return full_window[2:-6] + full_window[-3::2]
>>>>>>> REPLACE