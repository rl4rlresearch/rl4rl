MECHANISM: Robust top-k dual-statistic channel gating

HYPOTHESIS: Replacing the successful gate’s single-pixel maximum with the mean of its four strongest spatial responses will exceed 9,322 correct predictions by retaining localized evidence while reducing sensitivity to isolated activation noise.

INTENDED_EDIT: Restore Reference Design 3’s max-pooling backbone and identity-initialized refinement gate, but condition the gate on an equal blend of global mean and top-four spatial mean; retain its paired-view training, TTA, and calibrated temperature.

EVIDENCE: Reference Design 3’s mean/max gate achieved the best result at 9,322 correct, while unconstrained per-channel mean–peak mixing fell to 9,300; this motivates a parameter-free, robust refinement of the proven peak statistic rather than additional learned mixing.

<<<<<<< SEARCH
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(images + refined)
=======
        self.pointwise_norm = nn.BatchNorm2d(channels)

        gate_channels = max(channels // 9, 1)
        self.gate_down = nn.Conv2d(
            channels,
            gate_channels,
            kernel_size=1,
        )
        self.gate_up = nn.Conv2d(
            gate_channels,
            channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        average = F.adaptive_avg_pool2d(refined, output_size=1)
        top_values = refined.flatten(2).topk(4, dim=2).values
        robust_peak = top_values.mean(dim=2, keepdim=True).unsqueeze(-1)
        pooled = 0.5 * (average + robust_peak)
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
        return F.gelu(images + gate * refined)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(24 * 4, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(48 * 4, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SpatialRefinement(64),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.78
=======
        ).log() / 0.75317
>>>>>>> REPLACE