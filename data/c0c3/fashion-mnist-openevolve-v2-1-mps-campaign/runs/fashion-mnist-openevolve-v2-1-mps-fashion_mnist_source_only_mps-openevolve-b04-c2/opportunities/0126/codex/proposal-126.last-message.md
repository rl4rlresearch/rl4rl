MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.1977716064453125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410484313966.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.1978125 to 0.1977716064453125.

EVIDENCE: Momentum 0.197772216796875 achieved 9,295 correct, while 0.19777099609375 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1978125),
=======
            nn.BatchNorm1d(128, momentum=0.1977716064453125),
>>>>>>> REPLACE