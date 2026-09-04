MECHANISM: Lower-bound dense BatchNorm momentum bisection retry

HYPOTHESIS: BatchNorm1d momentum 0.1977734375 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028644546508789.

INTENDED_EDIT: Change only the classifier BatchNorm1d momentum from 0.19875 to 0.1977734375.

EVIDENCE: Momentum 0.1978125 achieved 9,295 correct with 0.20286445 cross-entropy, while 0.197734375 achieved 9,294 correct with lower 0.20286378 cross-entropy. Their exact midpoint remains unresolved because its prior verification timed out.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.19875),
=======
            nn.BatchNorm1d(128, momentum=0.1977734375),
>>>>>>> REPLACE