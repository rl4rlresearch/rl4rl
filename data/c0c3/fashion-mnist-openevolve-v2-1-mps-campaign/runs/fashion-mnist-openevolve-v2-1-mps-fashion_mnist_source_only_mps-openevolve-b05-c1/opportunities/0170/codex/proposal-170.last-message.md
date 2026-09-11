MECHANISM: Learned per-image subpixel feature registration

HYPOTHESIS: Replacing the assumption that fixed crop augmentation and ten-view TTA sufficiently handle translation with bounded, learned two-axis feature alignment will exceed 9,360 correct predictions by presenting the position-sensitive classifier with a more consistent spatial representation.

INTENDED_EDIT: Add an identity-initialized localization network that predicts a bounded continuous translation for each 7×7 feature map and resamples it before channel gating, statistics extraction, and classification; parameters increase from 247,546 to 249,400.

EVIDENCE: The verified model still relies on extensive translated-view averaging, while sparse diagonal crop augmentation reduced validation_correct to 9,340 and anti-aliased pooling did not finish. This motivates adaptive per-image registration rather than additional fixed views or another classifier-head modification.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.alignment_head = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(4 * 7 * 7, 8),
            nn.GELU(),
            nn.Linear(8, 2),
        )
        nn.init.zeros_(self.alignment_head[-1].weight)
        nn.init.zeros_(self.alignment_head[-1].bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        translations = 0.12 * torch.tanh(
            self.alignment_head(feature_map)
        )
        ones = torch.ones_like(translations[:, 0])
        zeros = torch.zeros_like(translations[:, 0])
        transform = torch.stack(
            (
                ones,
                zeros,
                translations[:, 0],
                zeros,
                ones,
                translations[:, 1],
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        sampling_grid = F.affine_grid(
            transform,
            feature_map.shape,
            align_corners=False,
        )
        feature_map = F.grid_sample(
            feature_map,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        spatial_mean = feature_map.mean(dim=(2, 3))
>>>>>>> REPLACE