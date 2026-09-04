MECHANISM: Lower-bound dense BatchNorm momentum interpolation

HYPOTHESIS: BatchNorm1d momentum 0.19875 will retain 9,295 correct predictions while reducing cross-entropy toward the lower value observed at momentum 0.1975, thereby exceeding the current validation_score.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.2 to 0.19875.

EVIDENCE: Momentum 0.2 achieved 9,295 correct with 0.2028835 cross-entropy, while 0.1975 achieved 9,294 correct with lower 0.2028617 cross-entropy; their midpoint directly probes whether the calibration benefit can be gained without losing a correct prediction.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.2),
=======
            nn.BatchNorm1d(128, momentum=0.19875),
>>>>>>> REPLACE