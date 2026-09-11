MECHANISM: Lower-bound dense BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.1977734375 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.2028644546508789.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.198125 to 0.1977734375.

EVIDENCE: Momentum 0.1978125 achieved 9,295 correct with 0.20286445 cross-entropy, while 0.197734375 achieved 9,294 correct with lower 0.20286378 cross-entropy; bisecting this boundary is the most informative remaining calibration test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.198125),
=======
            nn.BatchNorm1d(128, momentum=0.1977734375),
>>>>>>> REPLACE