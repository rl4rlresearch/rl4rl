MECHANISM: Near-prefix sparse-tail frame removal

HYPOTHESIS: Processing frames 2–23, 27, and 29 will achieve at least 85% validation accuracy with 24 recurrent steps by removing frame 25, which is temporally closest to the preserved dense prefix.

INTENDED_EDIT: Remove frame 25 from the successful 25-step schedule while retaining the dense prefix and the later tail samples at frames 27 and 29.

EVIDENCE: The 25-step schedule using frames 2–23, 25, 27, and 29 achieved 85.52%; removing frame 23 failed at 84.91% and removing frame 27 failed at 84.66%, indicating that frame 25 is the remaining direct ablation candidate for reaching 24 steps.

<<<<<<< SEARCH
            return full_window[:-6] + full_window[-5::2]
=======
            return full_window[:-6] + full_window[-3::2]
>>>>>>> REPLACE