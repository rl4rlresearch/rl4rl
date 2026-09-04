MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.0062336029052734375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.

EVIDENCE: Momentum 0.006233599853515625 retained 9,279 correct, while 0.00623360595703125 lost one prediction with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0065),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0062336029052734375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0065),
=======
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0062336029052734375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0065),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0062336029052734375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0065),
=======
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0062336029052734375),
>>>>>>> REPLACE