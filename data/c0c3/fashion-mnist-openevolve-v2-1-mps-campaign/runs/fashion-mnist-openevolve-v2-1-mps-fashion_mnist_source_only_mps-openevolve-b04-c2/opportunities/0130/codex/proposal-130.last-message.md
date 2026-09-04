MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.19777164459228515625 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410102844238.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.197772216796875 to 0.19777164459228515625.

EVIDENCE: Momentum 0.1977716827392578125 retained 9,295 correct, while 0.1977716064453125 produced 9,294 correct with lower cross-entropy; their exact midpoint is the most informative unresolved correctness-boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.197772216796875),
=======
            nn.BatchNorm1d(128, momentum=0.19777164459228515625),
>>>>>>> REPLACE