MECHANISM: Lower-bound dense BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.1978125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.198125.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.205 to 0.1978125.

EVIDENCE: Momentum 0.198125 achieved 9,295 correct with 0.2028672 cross-entropy, while 0.1975 achieved 9,294 correct with the lower 0.2028617 cross-entropy; bisecting this boundary is the most targeted test for further calibration improvement without losing correctness.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.205),
=======
            nn.BatchNorm1d(128, momentum=0.1978125),
>>>>>>> REPLACE