MECHANISM: Fine-grained batch-normalization horizon refinement

HYPOTHESIS: Setting all four batch-normalization momenta to 0.0063 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208620245.

INTENDED_EDIT: Raise every batch-normalization momentum from 0.007 to 0.0063 while preserving architecture, optimization, loss, schedule, and evaluation calibration.

EVIDENCE: Momentum increases from 0.006 to 0.0061 and 0.0062 retained 9,279 correct while monotonically reducing cross-entropy; 0.0065 lost one correct, making 0.0063 the next informative point toward that boundary.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.007)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.0063)
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
            nn.BatchNorm2d(32, momentum=0.0063),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0063),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0063),
>>>>>>> REPLACE