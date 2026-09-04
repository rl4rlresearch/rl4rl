MECHANISM: Per-channel adaptive mean–peak refinement gating

HYPOTHESIS: Learning a separate mean-versus-peak pooling mixture for each refinement channel will exceed Reference Design 3’s 9,322 correct predictions by preserving its successful initialization while adapting localized-feature sensitivity by channel.

INTENDED_EDIT: Restore the qualified refinement gate, initialize it to Reference Design 3’s equal mean/max pooling, make that mixture learnable per channel, and use its calibrated temperature. The model has 249,945 learned parameters.

EVIDENCE: Dual-statistic gating improved average-only gating from 9,319 to 9,322 correct; this motivates retaining both statistics while allowing channels representing diffuse and localized features to weight them differently.

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
        self.pool_mix = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        average = F.adaptive_avg_pool2d(refined, output_size=1)
        peak = F.adaptive_max_pool2d(refined, output_size=1)
        mixture = torch.sigmoid(self.pool_mix)
        pooled = average + mixture * (peak - average)
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