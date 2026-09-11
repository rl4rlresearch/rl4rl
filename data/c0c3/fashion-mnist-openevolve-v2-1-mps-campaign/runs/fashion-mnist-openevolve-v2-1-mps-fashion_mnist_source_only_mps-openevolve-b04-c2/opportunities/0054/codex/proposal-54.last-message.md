MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.00623369140625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the successful 0.00623359375 setting and the one-error 0.0062337890625 setting.

EVIDENCE: Momentum 0.00623359375 achieved 9,279 correct, while 0.0062337890625 lost one prediction despite lower cross-entropy; bisecting this remaining boundary is the most informative next probe.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.0062328125)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.00623369140625)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.0062328125),
=======
            nn.BatchNorm2d(32, momentum=0.00623369140625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.0062328125),
=======
            nn.BatchNorm2d(64, momentum=0.00623369140625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.0062328125),
=======
            nn.BatchNorm2d(128, momentum=0.00623369140625),
>>>>>>> REPLACE