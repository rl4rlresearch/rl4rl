MECHANISM: Accuracy-boundary batch-normalization refinement

HYPOTHESIS: Momentum 0.006234375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617843.

INTENDED_EDIT: Set all four batch-normalization momenta to 0.006234375, preserving every other setting.

EVIDENCE: Momentum 0.00623125 achieved 9,279 correct, while 0.0062375 lowered cross-entropy but lost one prediction; the prior midpoint attempt was malformed and therefore left this boundary point untested.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.00623125)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.006234375)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.00623125),
=======
            nn.BatchNorm2d(32, momentum=0.006234375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.00623125),
=======
            nn.BatchNorm2d(64, momentum=0.006234375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.00623125),
=======
            nn.BatchNorm2d(128, momentum=0.006234375),
>>>>>>> REPLACE