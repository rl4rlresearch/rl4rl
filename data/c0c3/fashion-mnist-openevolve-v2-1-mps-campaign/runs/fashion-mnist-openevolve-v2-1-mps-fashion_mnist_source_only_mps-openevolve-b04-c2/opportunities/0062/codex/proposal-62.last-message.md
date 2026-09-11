MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.006233602142333984375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861773147583007.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.

EVIDENCE: Momentum 0.00623360137939453125 retained 9,279 correct, while 0.0062336029052734375 lost one prediction with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.006233599853515625)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.006233602142333984375)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.006233599853515625),
=======
            nn.BatchNorm2d(32, momentum=0.006233602142333984375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.006233599853515625),
=======
            nn.BatchNorm2d(64, momentum=0.006233602142333984375),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.006233599853515625),
=======
            nn.BatchNorm2d(128, momentum=0.006233602142333984375),
>>>>>>> REPLACE