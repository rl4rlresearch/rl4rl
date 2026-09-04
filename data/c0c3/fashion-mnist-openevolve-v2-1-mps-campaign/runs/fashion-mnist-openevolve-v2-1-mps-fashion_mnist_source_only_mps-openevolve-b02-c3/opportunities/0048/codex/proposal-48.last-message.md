MECHANISM: Shared dual-path mean–peak refinement gating

HYPOTHESIS: Processing mean and maximum responses separately through the shared nonlinear gate will exceed Reference Design 3’s 9,322 correct predictions while retaining its regularized parameter count.

INTENDED_EDIT: Restore the qualified max-pooling backbone and identity-initialized 64→7→64 gate, but average the separately transformed mean and peak descriptors immediately before gate projection; retain temperature 0.75317.

EVIDENCE: Equal mean/max gating achieved the best verified result of 9,322 correct, whereas average-only reached 9,319 and learned per-channel mixing fell to 9,300; preserving both statistics while separating them only across the shared nonlinearity tests greater expressiveness without extra parameters.

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
        mean_descriptor = F.adaptive_avg_pool2d(refined, output_size=1)
        peak_descriptor = F.adaptive_max_pool2d(refined, output_size=1)
        gate_features = 0.5 * (
            F.gelu(self.gate_down(mean_descriptor))
            + F.gelu(self.gate_down(peak_descriptor))
        )
        gate = 2.0 * torch.sigmoid(self.gate_up(gate_features))
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