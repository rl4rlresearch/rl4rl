MECHANISM: Slowly accumulated dense-feature batch normalization

HYPOTHESIS: BatchNorm1d with the feature extractor’s verified 0.00623359375 momentum will exceed 9,288 correct predictions by retaining the successful classifier conditioning while producing less noisy evaluation statistics than the default 0.1 momentum.

INTENDED_EDIT: Add BatchNorm1d after the first classifier linear layer, using the same low running-statistics momentum as the convolutional normalization layers.

EVIDENCE: Default-momentum dense BatchNorm achieved the best available result at 9,288 correct, while every BatchNorm2d layer in that successful feature extractor uses momentum 0.00623359375; applying that verified statistic-accumulation rate isolates whether more stable dense running statistics improve generalization.

<<<<<<< SEARCH
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
=======
            nn.Linear(128 * 3 * 3, 128),
            nn.BatchNorm1d(128, momentum=0.00623359375),
            nn.GELU(),
            nn.Dropout(p=0.05),
>>>>>>> REPLACE