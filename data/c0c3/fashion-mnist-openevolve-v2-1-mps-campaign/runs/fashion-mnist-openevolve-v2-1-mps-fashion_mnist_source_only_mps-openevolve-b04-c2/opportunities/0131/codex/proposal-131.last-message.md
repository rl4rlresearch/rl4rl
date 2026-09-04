MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.197771625518798828125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.19777191162109375 to 0.197771625518798828125.

EVIDENCE: Momentum 0.19777164459228515625 achieved 9,295 correct with the best observed cross-entropy, while 0.1977716064453125 produced 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved correctness-boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.19777191162109375),
=======
            nn.BatchNorm1d(128, momentum=0.197771625518798828125),
>>>>>>> REPLACE