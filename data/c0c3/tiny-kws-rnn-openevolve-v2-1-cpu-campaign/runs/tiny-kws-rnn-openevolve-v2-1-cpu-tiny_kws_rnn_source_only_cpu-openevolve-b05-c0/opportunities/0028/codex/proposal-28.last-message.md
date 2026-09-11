MECHANISM: Dense-prefix trimming with preserved sparse tail coverage

HYPOTHESIS: Processing frames 2–22, 25, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and total inference MACs by approximately 4%.

INTENDED_EDIT: Remove frame 23 from the dense prefix while preserving the three late sampled frames and the full temporal endpoint.

EVIDENCE: The 25-step schedule spanning frames 2–29 achieved 85.52%, whereas the contiguous 25-step schedule ending at frame 26 achieved only 83.31%; this indicates late temporal coverage should be preserved while testing one fewer dense-prefix frame.

<<<<<<< SEARCH
            return full_window[:-6] + full_window[-5::2]
=======
            return full_window[:-7] + full_window[-5::2]
>>>>>>> REPLACE