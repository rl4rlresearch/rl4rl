MECHANISM: Lower-bound dense BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.19775390625 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286411476135255.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.19775390625.

EVIDENCE: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.197734375 achieved 9,294 correct with lower 0.20286377944946288 cross-entropy; bisecting this unresolved boundary is the most targeted test for improving calibration without losing correctness.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1975),
=======
            nn.BatchNorm1d(128, momentum=0.19775390625),
>>>>>>> REPLACE