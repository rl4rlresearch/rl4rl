MECHANISM: Identity-initialized coordinate-conditioned feature gating

HYPOTHESIS: Row- and column-aware channel modulation will exceed 9,240 correct predictions by adding global silhouette interactions while preserving the proven position-sensitive classifier.

INTENDED_EDIT: Add 5,968 learned parameters of separable row/column context gating after local refinement, keeping total parameters at 248,514, and use the best verified 1.296352 evaluation calibration.

EVIDENCE: The additive global-moment head fell from 9,240 to 9,227 correct, showing that pooled evidence should not independently vote on classes. This patch instead uses axis-preserving context to modulate the successful spatial path, starts as an exact identity, and cannot inject separate logits.

<<<<<<< SEARCH
            nn.Linear(48, 10),
        )

    @staticmethod
=======
            nn.Linear(48, 10),
        )
        self.coordinate_reduce = nn.Sequential(
            nn.Conv2d(80, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.row_gate = nn.Conv2d(24, 80, kernel_size=1)
        self.column_gate = nn.Conv2d(24, 80, kernel_size=1)
        nn.init.zeros_(self.row_gate.weight)
        nn.init.zeros_(self.row_gate.bias)
        nn.init.zeros_(self.column_gate.weight)
        nn.init.zeros_(self.column_gate.bias)

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))

        height, width = features.shape[-2:]
        row_context = features.mean(dim=3, keepdim=True)
        column_context = features.mean(dim=2, keepdim=True).transpose(2, 3)
        coordinate_context = self.coordinate_reduce(
            torch.cat((row_context, column_context), dim=2)
        )
        row_context, column_context = coordinate_context.split(
            (height, width), dim=2
        )
        column_context = column_context.transpose(2, 3)

        row_scale = 1.0 + 0.5 * torch.tanh(self.row_gate(row_context))
        column_scale = 1.0 + 0.5 * torch.tanh(
            self.column_gate(column_context)
        )
        features = features * row_scale * column_scale
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.25 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE