MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.0062336027622222900390625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.

INTENDED_EDIT: Set all four batch-normalization momenta to the untested midpoint between the highest successful setting and the nearest one-error setting.

EVIDENCE: Momentum 0.00623360271453857421875 retained 9,279 correct, while 0.006233602809906005859375 produced 9,278 with lower cross-entropy; bisecting this remaining boundary is the most informative next probe.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.006233599853515625)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.0062336027622222900390625)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.006233599853515625),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0062336027622222900390625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006233599853515625),
=======
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0062336027622222900390625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006233599853515625),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0062336027622222900390625),
>>>>>>> REPLACE