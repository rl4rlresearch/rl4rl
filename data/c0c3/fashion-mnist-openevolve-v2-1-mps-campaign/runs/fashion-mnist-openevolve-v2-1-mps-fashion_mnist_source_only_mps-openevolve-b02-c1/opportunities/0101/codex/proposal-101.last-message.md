MECHANISM: Budget-saturating normalized classifier head

HYPOTHESIS: Widening the inexpensive dense head to 147 units and normalizing its activations will exceed 9,286 validation-correct predictions while staying below the 250,000-parameter ceiling and verification time limit.

INTENDED_EDIT: Preserve the proven convolutional, pooling, training, averaging, and TTA paths; use the remaining parameter budget to widen the classifier and add BatchNorm, producing 249,601 learned parameters.

EVIDENCE: Changing terminal pooling regressed to 9,234, while added convolutional refinement timed out; this motivates preserving the successful feature extractor and adding low-cost capacity only in the dense head.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
>>>>>>> REPLACE