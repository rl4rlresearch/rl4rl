MECHANISM: Accuracy-boundary batch-normalization refinement

HYPOTHESIS: Momentum 0.006234375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617843.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between 0.00623125 and 0.0062375, preserving every other setting.

EVIDENCE: Momentum 0.00623125 achieved 9,279 correct, while 0.0062375 reduced cross-entropy but lost one prediction; bisecting this narrow boundary is the most informative next test.

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.0065),
=======
            nn.BatchNorm2d(32, momentum=0.006234375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.0065),
=======
            nn.BatchNorm2d(64, momentum=0.006234375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.0065),
            nn.GELU(),
            nn.Conv2d(
=======
            nn.BatchNorm2d(128, momentum=0.006234375),
            nn.GELU(),
            nn.Conv2d(
>>>>>>> REPLACE

<<<<<<< SEARCH
            )
            nn.BatchNorm2d(128, momentum=0.0065),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            )
            nn.BatchNorm2d(128, momentum=0.006234375),
            nn.GELU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE