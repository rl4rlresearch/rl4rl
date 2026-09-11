MECHANISM: Decision-boundary BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.19777099609375 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.20286411476135255.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.19777099609375.

EVIDENCE: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.1977685546875 achieved 9,294 correct with lower 0.2028640724182129 cross-entropy; their exact midpoint is the most informative unresolved boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1975),
=======
            nn.BatchNorm1d(128, momentum=0.19777099609375),
>>>>>>> REPLACE