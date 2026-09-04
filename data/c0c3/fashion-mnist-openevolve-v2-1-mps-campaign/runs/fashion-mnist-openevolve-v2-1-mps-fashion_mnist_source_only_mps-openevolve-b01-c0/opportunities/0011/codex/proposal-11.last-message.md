MECHANISM: Near-ceiling dense bottleneck expansion

HYPOTHESIS: Expanding the dense bottleneck from 48 to 58 units will exceed 9,265 correct predictions by using the remaining parameter budget to improve class separation without altering the successful augmentation and optimization procedure.

INTENDED_EDIT: Increase the classifier bottleneck width to 58 units, raising learned parameters from 216,346 to 247,836 while retaining all other settings.

EVIDENCE: Reallocating dense-head capacity into a fifth convolution reduced validation correct from 9,177 to 9,157; preserving the successful convolutional extractor and adding capacity to the dense head is therefore the most direct untested use of the 250,000-parameter ceiling.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.BatchNorm1d(58),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(58, 10),
        )
>>>>>>> REPLACE