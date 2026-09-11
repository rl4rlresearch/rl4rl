MECHANISM: Incremental trailing-boundary trimming

HYPOTHESIS: Processing frames 2–27 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.7% versus the verified 27-step model.

INTENDED_EDIT: Omit one additional trailing frame, producing a contiguous 26-frame schedule while preserving the model and training procedure.

EVIDENCE: Frames 2–28 achieved 86.26% accuracy after frames 2–29 achieved 86.50%, indicating that incremental removal from the less-informative recording tail has preserved a meaningful margin above the target.

<<<<<<< SEARCH
        if available_frames >= 6:
            return list(range(2, available_frames - 3))
=======
        if available_frames >= 6:
            return list(range(2, available_frames - 4))
>>>>>>> REPLACE