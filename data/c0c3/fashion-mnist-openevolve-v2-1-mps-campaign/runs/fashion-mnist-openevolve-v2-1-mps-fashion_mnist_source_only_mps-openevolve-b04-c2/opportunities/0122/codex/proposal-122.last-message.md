MECHANISM: Lower-bound dense BatchNorm momentum bisection

HYPOTHESIS: BatchNorm1d momentum 0.1977685546875 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286411476135255.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.1978125 to 0.1977685546875.

EVIDENCE: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.197763671875 achieved 9,294 correct with the lower 0.2028640281677246 cross-entropy; their midpoint is the most targeted unresolved boundary test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.1978125),
=======
            nn.BatchNorm1d(128, momentum=0.1977685546875),
>>>>>>> REPLACE