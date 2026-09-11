MECHANISM: Dual-statistic mean–peak refinement gating

HYPOTHESIS: Restoring Reference Design 3 exactly will improve validation correctness from 9,315 to approximately 9,322 while remaining below the 250,000-parameter ceiling.

INTENDED_EDIT: Add the identity-initialized 64→7→64 refinement gate, condition it on an equal blend of global mean and maximum responses, and use the reference design’s calibrated temperature of 0.75317.

EVIDENCE: Reference Design 3 achieved the best verified result—9,322 correct with 0.1904878 cross-entropy—while average-only gating reached 9,319 and a 60/40 mean–maximum blend fell to 9,297.

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
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
        return F.gelu(images + gate * refined)
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.753
=======
        ).log() / 0.75317
>>>>>>> REPLACE