MECHANISM: Content-adaptive mixed-statistic downsampling

HYPOTHESIS: Replacing fixed max pooling with locally gated interpolation between mean and maximum responses will exceed 9,206 correct predictions by preserving distributed stroke evidence without sacrificing salient features; verified 1.05× sharpening will improve cross-entropy if correctness ties.

INTENDED_EDIT: Challenge the shared assumption that one hard-coded pooling statistic suits every feature and location. Each residual stage instead learns a depthwise, image-conditioned 2×2 pooling gate, initialized near max pooling; retain the validated evaluation sharpening.

EVIDENCE: Residual refinement improved correctness from 9,168 to 9,206, while the late global-moment branch reached only 9,201, indicating that richer information should be integrated within the spatial hierarchy. The extensive translation augmentation and shifted-view ensemble further identify downsampling sensitivity as load-bearing. Reference Design 3 verified that 1.05× sharpening preserves argmax predictions while lowering cross-entropy.

<<<<<<< SEARCH
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))
=======
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool_gate = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            groups=out_channels,
        )
        nn.init.zeros_(self.pool_gate.weight)
        nn.init.constant_(self.pool_gate.bias, 2.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        features = F.silu(features + refinement, inplace=True)
        pooled_mean = F.avg_pool2d(features, 2)
        pooled_max = F.max_pool2d(features, 2)
        gate = torch.sigmoid(self.pool_gate(features))
        return pooled_mean + gate * (pooled_max - pooled_mean)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE