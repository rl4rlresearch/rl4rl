MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.0062336181640625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.

EVIDENCE: Momentum 0.00623359375 achieved 9,279 correct, while 0.006233642578125 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.0065),
=======
            nn.BatchNorm2d(32, momentum=0.0062336181640625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.0065),
=======
            nn.BatchNorm2d(64, momentum=0.0062336181640625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.0065),
=======
            nn.BatchNorm2d(128, momentum=0.0062336181640625),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.0065),
=======
            nn.BatchNorm2d(128, momentum=0.0062336181640625),
>>>>>>> REPLACE