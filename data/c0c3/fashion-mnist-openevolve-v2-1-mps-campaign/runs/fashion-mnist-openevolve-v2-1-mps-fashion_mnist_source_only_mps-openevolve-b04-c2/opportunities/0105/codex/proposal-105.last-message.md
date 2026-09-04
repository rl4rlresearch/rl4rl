MECHANISM: Faster dense-feature running-statistic adaptation

HYPOTHESIS: BatchNorm1d momentum 0.2 will exceed 9,288 correct predictions by tracking the rapidly evolving classifier representation more closely than the best verified default momentum of 0.1.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from its default 0.1 to 0.2 while preserving the architecture, optimization, loss, and evaluation calibration.

EVIDENCE: Classifier BatchNorm momentum 0.1 achieved 9,288 correct, versus 9,284 at momentum 0.00623359375 and 9,281 with live validation-batch statistics; this makes faster—but still running-averaged—statistics the clearest untested direction.

<<<<<<< SEARCH
            nn.BatchNorm1d(128),
=======
            nn.BatchNorm1d(128, momentum=0.2),
>>>>>>> REPLACE