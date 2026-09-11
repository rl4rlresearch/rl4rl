MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.197771759033203125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410484313966.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.197771759033203125.

EVIDENCE: Momentum 0.19777191162109375 achieved 9,295 correct, while 0.1977716064453125 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1975),
=======
            nn.BatchNorm1d(128, momentum=0.197771759033203125),
>>>>>>> REPLACE