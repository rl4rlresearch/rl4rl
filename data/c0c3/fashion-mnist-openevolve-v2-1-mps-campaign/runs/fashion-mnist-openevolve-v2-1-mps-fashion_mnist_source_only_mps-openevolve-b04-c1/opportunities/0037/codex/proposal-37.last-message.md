MECHANISM: Identity-initialized grouped spatial refinement

HYPOTHESIS: Adding a stable grouped 3×3 residual block at 7×7 resolution will exceed 9,286 correct predictions by refining spatial relationships before the classifier, while remaining below the parameter ceiling.

INTENDED_EDIT: Add an 18,560-parameter, identity-initialized grouped-convolution residual block after the second pooling stage; retain the proven optimizer, loss, augmentation, EMA, and evaluation calibration.

EVIDENCE: Widening the classifier reduced correct predictions to 9,254, while temperature calibration has saturated without changing predictions; this tests additional spatial feature processing instead of more classifier capacity.

<<<<<<< SEARCH
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
=======
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                groups=2,
                bias=False,
            ),
            nn.BatchNorm2d(64),
        )
        nn.init.zeros_(self.refine[-1].weight)
        self.pool = nn.MaxPool2d(2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        return self.classifier(features)
=======
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
>>>>>>> REPLACE