MECHANISM: Input-conditioned squeeze-excitation residual channels

HYPOTHESIS: Adding lightweight channel gates while retaining the qualified half-weight unshifted TTA will exceed 9,208 correct predictions by adapting feature-channel importance to each image without the runtime cost that prevented learned mixed pooling from finishing.

INTENDED_EDIT: Add squeeze-excitation gating to every residual stage, shrink the dense hidden layer from 48 to 44 to remain below the parameter ceiling, and use the best verified 0.5 unshifted-view weighting.

EVIDENCE: Reference Design 1 achieved the best result, 9,208 correct, through TTA weighting alone; subsequent aggregation changes did not progress, while learned mixed pooling timed out. This challenges the shared assumption of sample-independent channel importance using a much cheaper dynamic representation mechanism.

<<<<<<< SEARCH
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))
=======
        self.norm2 = nn.BatchNorm2d(out_channels)
        gate_channels = max(4, out_channels // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, gate_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        features = F.silu(features + refinement, inplace=True)
        features = features * (2.0 * self.channel_gate(features))
        return self.pool(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
=======
            nn.Linear(96 * 3 * 3, 44),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(44, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )
=======
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.5
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.5,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE