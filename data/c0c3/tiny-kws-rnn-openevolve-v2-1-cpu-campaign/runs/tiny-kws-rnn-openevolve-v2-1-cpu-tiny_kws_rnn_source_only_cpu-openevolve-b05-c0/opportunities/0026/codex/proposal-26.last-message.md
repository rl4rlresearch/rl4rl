MECHANISM: Incremental trailing-boundary trimming

HYPOTHESIS: Processing frames 2–26 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.8% versus the verified 26-step model.

INTENDED_EDIT: Omit one additional trailing frame, producing a contiguous 25-frame schedule while preserving model capacity and training procedure.

EVIDENCE: Frames 2–27 achieved 85.15% accuracy after successive trailing trims at 28 and 27 endpoints also met the target; this is the smallest structural reduction that directly tests the remaining temporal boundary, though the narrow margin makes it a high-information threshold test.

<<<<<<< SEARCH
        if available_frames >= 6:
            return list(range(2, available_frames - 4))
=======
        if available_frames >= 7:
            return list(range(2, available_frames - 5))
>>>>>>> REPLACE