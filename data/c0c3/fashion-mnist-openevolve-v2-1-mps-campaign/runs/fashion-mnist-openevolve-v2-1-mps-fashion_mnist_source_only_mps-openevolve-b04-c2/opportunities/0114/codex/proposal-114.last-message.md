MECHANISM: Lower-bound dense BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.198125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.19875.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.2025 to 0.198125.

EVIDENCE: Momentum 0.19875 achieved 9,295 correct with 0.2028726 cross-entropy, while 0.1975 achieved 9,294 correct with a lower 0.2028617 cross-entropy; bisecting this boundary tests whether more of the calibration benefit is available without losing correctness.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.2025),
=======
            nn.BatchNorm1d(128, momentum=0.198125),
>>>>>>> REPLACE