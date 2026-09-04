MECHANISM: Incremental trailing-boundary trimming

HYPOTHESIS: Processing frames 2–28 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.6% versus the verified 28-frame model.

INTENDED_EDIT: Preserve the informative starting frame and omit one additional trailing frame, producing a contiguous 27-frame schedule.

EVIDENCE: Frames 2–29 achieved 86.50% accuracy at 28 steps, outperforming the 29-step final-frame schedule; this suggests earlier command-onset information is more valuable than the recording tail and motivates the smallest further trailing trim.

<<<<<<< SEARCH
        if available_frames >= 6:
            return list(range(2, available_frames - 2))
=======
        if available_frames >= 6:
            return list(range(2, available_frames - 3))
>>>>>>> REPLACE