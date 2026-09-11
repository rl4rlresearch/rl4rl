MECHANISM: Zero-initialized class-specific local-evidence pooling

HYPOTHESIS: A residual 7×7 class-evidence head using log-mean-exp spatial pooling will exceed 9,284 correct predictions by learning translation-robust part presence, while preserving the validated classifier exactly at initialization.

INTENDED_EDIT: Add a zero-initialized 810-parameter class-map branch before spatial aggregation, fuse its pooled logits with the layout-sensitive classifier, and use Reference Design 3’s strongest validated TTA weights.

EVIDENCE: TTA boundary searches plateaued at 9,284 correct, while content-conditioned downsampling fell to 9,280; this retains all validated representation and pooling behavior but challenges the shared assumption that predictions must come solely from the flattened spatial head.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
        self.part_weight = nn.Parameter(torch.zeros(10, 80, 1, 1))
        self.part_bias = nn.Parameter(torch.zeros(10))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))

        part_maps = F.conv2d(features, self.part_weight, self.part_bias)
        part_logits = torch.logsumexp(part_maps.flatten(2), dim=2)
        part_logits = part_logits - math.log(
            part_maps.shape[-2] * part_maps.shape[-1]
        )

        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features) + part_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = 0.3625 * self._flip_average(images)
=======
        logits = 0.3634375 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.07000
=======
                    elif delta_y == 0:
                        weight = 0.06953125
>>>>>>> REPLACE