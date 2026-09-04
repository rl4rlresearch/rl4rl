MECHANISM: Tail-endpoint ablation with preserved informative mid-tail coverage

HYPOTHESIS: Processing frames 2–23 and 27 will retain at least 85% validation accuracy while reducing execution from 24 to 23 recurrent steps and total inference MACs by approximately 4.2%.

INTENDED_EDIT: Remove frame 29 from the current schedule while preserving the dense prefix and frame 27.

EVIDENCE: Removing frame 27 previously failed at 84.66% despite retaining frame 29, whereas removing frame 25 succeeded at 85.28%; this identifies frame 27 as the stronger late sample and makes frame 29 the most informative remaining ablation.

<<<<<<< SEARCH
        if available_frames >= 8:
            full_window = list(range(2, available_frames - 2))
            return full_window[:-6] + full_window[-3::2]
=======
        if available_frames >= 11:
            full_window = list(range(2, available_frames - 2))
            return full_window[:-6] + [full_window[-3]]
>>>>>>> REPLACE