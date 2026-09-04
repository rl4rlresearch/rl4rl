MECHANISM: Evaluation-time dense-feature normalization

HYPOTHESIS: Using current validation-batch statistics in the classifier BatchNorm will eliminate its mismatch with tail-averaged weights and exceed 9,311 correct predictions without additional parameters or training computation.

INTENDED_EDIT: Disable running-stat tracking only for the 147-unit classifier BatchNorm, preserving the verified feature extractor, optimizer, augmentation, averaging, and weighted-logit TTA.

EVIDENCE: The tail-averaged normalized head achieved the best 9,311 correct, while reducing averaging regressed to 9,307; prior attempts to align BatchNorm buffers timed out, leaving the identified parameter/statistics mismatch unresolved.

<<<<<<< SEARCH
            nn.BatchNorm1d(147),
=======
            nn.BatchNorm1d(147, track_running_stats=False),
>>>>>>> REPLACE