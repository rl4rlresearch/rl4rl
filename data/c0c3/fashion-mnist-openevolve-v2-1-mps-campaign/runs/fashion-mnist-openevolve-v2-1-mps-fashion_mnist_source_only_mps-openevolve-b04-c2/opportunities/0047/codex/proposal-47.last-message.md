MECHANISM: Accuracy-boundary batch-normalization refinement

HYPOTHESIS: Momentum 0.00623125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208618358.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the successful 0.006225 setting and the one-error 0.0062375 setting, preserving every other setting.

EVIDENCE: Momentum 0.006225 achieved 9,279 correct, while 0.0062375 lowered cross-entropy but lost one prediction; bisecting this narrow boundary is the most informative next test.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.007)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.00623125)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.007),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.007),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.007),
=======
            nn.BatchNorm2d(32, momentum=0.00623125),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.00623125),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.00623125),
>>>>>>> REPLACE