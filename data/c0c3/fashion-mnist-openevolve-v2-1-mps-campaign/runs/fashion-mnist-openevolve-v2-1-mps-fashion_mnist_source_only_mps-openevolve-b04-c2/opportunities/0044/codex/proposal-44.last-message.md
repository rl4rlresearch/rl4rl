MECHANISM: Accuracy-boundary batch-normalization refinement

HYPOTHESIS: Momentum 0.00625 will retain 9,279 correct predictions while lowering cross-entropy below the 0.0062 result of 0.208620245.

INTENDED_EDIT: Set all four batch-normalization momenta from 0.0065 to 0.00625, preserving every other setting.

EVIDENCE: Momentum 0.0062 achieved 9,279 correct, while 0.0063 lowered cross-entropy to 0.208612660 but lost one correct prediction; their midpoint directly probes the apparent accuracy boundary.

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.0065),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0065),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0065),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0065),
=======
            nn.BatchNorm2d(32, momentum=0.00625),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.00625),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.00625),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.00625),
>>>>>>> REPLACE