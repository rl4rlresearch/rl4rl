MECHANISM: More responsive bottleneck running-statistics estimation

HYPOTHESIS: Increasing bottleneck BatchNorm momentum from 0.1 to 0.2 will exceed 9,314 correct predictions by aligning evaluation statistics more closely with the late-training feature distribution.

INTENDED_EDIT: Set only the 30-feature BatchNorm1d momentum to 0.2, preserving all other architecture, optimization, augmentation, and inference behavior.

EVIDENCE: Bottleneck BatchNorm raised correctness from 9,249 to 9,314, while reducing its momentum to 0.05 preserved correctness but worsened cross-entropy from 0.1920517 to 0.1921869; this supports testing the opposite, more-responsive direction.

<<<<<<< SEARCH
            nn.BatchNorm1d(30),
=======
            nn.BatchNorm1d(30, momentum=0.2),
>>>>>>> REPLACE