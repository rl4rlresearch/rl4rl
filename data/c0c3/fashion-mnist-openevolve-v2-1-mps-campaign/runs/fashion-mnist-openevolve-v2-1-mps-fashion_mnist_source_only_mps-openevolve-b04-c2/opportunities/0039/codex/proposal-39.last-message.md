MECHANISM: Near-optimal batch-normalization averaging horizon

HYPOTHESIS: Setting all four batch-normalization momenta to 0.006 will reproduce the verified 9,279 correct predictions and exceed the current 9,273.

INTENDED_EDIT: Apply momentum=0.006 to every batch-normalization layer while preserving architecture, optimization, loss, schedule, and evaluation calibration.

EVIDENCE: Reference Design 2 achieved the best available result—9,279 correct—with this isolated setting; neighboring momenta 0.005 and 0.007 each reached only 9,278.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.006)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
=======
            nn.BatchNorm2d(32, momentum=0.006),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64),
=======
            nn.BatchNorm2d(64, momentum=0.006),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128),
=======
            nn.BatchNorm2d(128, momentum=0.006),
>>>>>>> REPLACE