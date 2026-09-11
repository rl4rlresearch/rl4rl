MECHANISM: Lower-bound dense BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.19765625 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.2028644546508789.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.19765625.

EVIDENCE: Momentum 0.1978125 achieved 9,295 correct at 0.20286445 cross-entropy, while 0.1975 achieved 9,294 correct at the lower 0.20286175; bisecting this boundary is the most targeted test for improved calibration without losing correctness.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1975),
=======
            nn.BatchNorm1d(128, momentum=0.19765625),
>>>>>>> REPLACE