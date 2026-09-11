MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.0062328125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617843.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the successful 0.00623125 setting and the one-error 0.006234375 setting.

EVIDENCE: Momentum 0.00623125 achieved 9,279 correct, while 0.006234375 lost one prediction despite lower cross-entropy; bisecting their boundary is the most informative next test.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.006225)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.0062328125)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.006225),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0062328125),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006225),
=======
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0062328125),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006225),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0062328125),
>>>>>>> REPLACE