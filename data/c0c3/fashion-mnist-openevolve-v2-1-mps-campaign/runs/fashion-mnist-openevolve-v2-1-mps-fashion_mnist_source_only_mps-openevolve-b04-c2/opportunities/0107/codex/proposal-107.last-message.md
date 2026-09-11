MECHANISM: Fine-grained dense BatchNorm momentum tuning

HYPOTHESIS: BatchNorm1d momentum 0.25 will exceed 9,295 correct predictions by balancing the superior adaptation at momentum 0.2 against the slight degradation observed at 0.3.

INTENDED_EDIT: Replace validation-batch statistics with running statistics at momentum 0.25, preserving all other architecture, optimization, loss, and calibration settings.

EVIDENCE: Momentum 0.2 achieved the best result of 9,295 correct, while 0.1 achieved 9,288 and 0.3 achieved 9,292; testing their unmeasured midpoint most directly refines the apparent optimum.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, track_running_stats=False),
=======
            nn.BatchNorm1d(128, momentum=0.25),
>>>>>>> REPLACE