MECHANISM: Lower-bound dense BatchNorm momentum bisection retry

HYPOTHESIS: BatchNorm1d momentum 0.1977685546875 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.20286411476135255.

INTENDED_EDIT: Decrease only the classifier BatchNorm1d momentum from 0.198125 to 0.1977685546875.

EVIDENCE: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.197763671875 achieved 9,294 correct with lower 0.2028640281677246 cross-entropy. Their exact midpoint remains unresolved because its previous verification timed out.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.198125),
=======
            nn.BatchNorm1d(128, momentum=0.1977685546875),
>>>>>>> REPLACE