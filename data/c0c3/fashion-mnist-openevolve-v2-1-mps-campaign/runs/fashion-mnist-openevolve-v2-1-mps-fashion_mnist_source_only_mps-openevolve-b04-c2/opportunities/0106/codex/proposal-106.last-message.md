MECHANISM: Faster dense-feature running-statistic adaptation

HYPOTHESIS: BatchNorm1d momentum 0.3 will exceed 9,295 correct predictions by tracking the final classifier representation more closely than momentum 0.2.

INTENDED_EDIT: Increase only the classifier BatchNorm1d momentum from 0.00623359375 to 0.3, preserving all other architecture, optimization, loss, and calibration settings.

EVIDENCE: Validation correctness increased from 9,284 at momentum 0.00623359375 to 9,288 at 0.1 and 9,295 at 0.2, making a further increase the most direct test of the observed adaptation trend.

<<<<<<< SEARCH
            nn.BatchNorm1d(128, momentum=0.00623359375),
=======
            nn.BatchNorm1d(128, momentum=0.3),
>>>>>>> REPLACE