MECHANISM: Fine-grained dense BatchNorm momentum interpolation

HYPOTHESIS: BatchNorm1d momentum 0.1975 will exceed 9,295 correct predictions by preserving the stronger adaptation at 0.2 while slightly reducing running-statistic noise toward 0.19.

INTENDED_EDIT: Change only the classifier BatchNorm1d momentum from 0.25 to 0.1975.

EVIDENCE: Momentum 0.2 achieved the best score with 9,295 correct, while 0.19 achieved 9,293; testing their midpoint probes the remaining unmeasured interval around the observed optimum.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.25),
=======
            nn.BatchNorm1d(128, momentum=0.1975),
>>>>>>> REPLACE