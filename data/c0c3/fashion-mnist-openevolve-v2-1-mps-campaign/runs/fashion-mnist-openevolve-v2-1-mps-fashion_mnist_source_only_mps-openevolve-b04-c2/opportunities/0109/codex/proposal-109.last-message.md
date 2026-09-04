MECHANISM: Fine-grained dense BatchNorm momentum refinement

HYPOTHESIS: BatchNorm1d momentum 0.21 will exceed 9,295 correct predictions by retaining the rapid statistic adaptation at 0.2 while testing the narrow interval before performance declines at 0.25.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from 0.2 to 0.21, preserving the best architecture, optimizer, loss, schedule, and evaluation calibration.

EVIDENCE: Momentum 0.2 achieved the best result with 9,295 correct, while 0.19 achieved 9,293 and both 0.25 and 0.3 achieved 9,292; the evidence brackets a narrow optimum near 0.2 and makes a small upward refinement the most informative next test.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.2),
=======
            nn.BatchNorm1d(128, momentum=0.21),
>>>>>>> REPLACE