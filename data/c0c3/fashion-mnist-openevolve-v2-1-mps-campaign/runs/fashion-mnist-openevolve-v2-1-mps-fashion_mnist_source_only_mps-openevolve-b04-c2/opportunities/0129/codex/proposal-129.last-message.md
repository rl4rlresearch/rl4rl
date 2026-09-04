MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.1977716827392578125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410102844238.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.197771759033203125 to 0.1977716827392578125.

EVIDENCE: Momentum 0.197771759033203125 achieved 9,295 correct with 0.20286410102844238 cross-entropy, while 0.1977716064453125 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.197771759033203125),
=======
            nn.BatchNorm1d(128, momentum=0.1977716827392578125),
>>>>>>> REPLACE