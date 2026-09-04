MECHANISM: Zero-initialized pointwise channel-refinement residual

HYPOTHESIS: A compute-efficient nonlinear channel-mixing residual will exceed 9,284 correct predictions while retaining the validated model’s initial computation and staying below the parameter ceiling.

INTENDED_EDIT: Add a zero-initialized 80→28→80 pointwise residual before aggregation, increasing learned parameters from 245,152 to 249,848 while preserving the existing training and evaluation paths.

EVIDENCE: Channel reweighting and channel-specific pooling regressed, while spatial attention exceeded the verification time limit; this tests inexpensive nonlinear feature refinement rather than attention or pooling changes, with zero initialization protecting the established baseline.

<<<<<<< SEARCH
            nn.BatchNorm2d(80),
        )
        self.aggregate_local = nn.Sequential(
=======
            nn.BatchNorm2d(80),
        )
        self.channel_refine = nn.Sequential(
            nn.Conv2d(80, 28, kernel_size=1, bias=False),
            nn.BatchNorm2d(28),
            nn.GELU(),
            nn.Conv2d(28, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
        )
        nn.init.zeros_(self.channel_refine[-1].weight)
        self.aggregate_local = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
=======
        features = self.features(self._image_basis(images))
        features = F.gelu(
            features
            + self.refine(features)
            + self.channel_refine(features)
        )
        features = torch.cat(
>>>>>>> REPLACE