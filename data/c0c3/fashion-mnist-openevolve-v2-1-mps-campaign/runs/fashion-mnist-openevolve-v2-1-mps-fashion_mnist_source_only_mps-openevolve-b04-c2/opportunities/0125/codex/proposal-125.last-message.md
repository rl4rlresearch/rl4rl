MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.197772216796875 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286411476135255.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.1977734375 to 0.197772216796875.

EVIDENCE: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.19777099609375 achieved 9,294 correct with lower 0.20286409683227538 cross-entropy; their exact midpoint is the most informative unresolved boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1977734375),
=======
            nn.BatchNorm1d(128, momentum=0.197772216796875),
>>>>>>> REPLACE