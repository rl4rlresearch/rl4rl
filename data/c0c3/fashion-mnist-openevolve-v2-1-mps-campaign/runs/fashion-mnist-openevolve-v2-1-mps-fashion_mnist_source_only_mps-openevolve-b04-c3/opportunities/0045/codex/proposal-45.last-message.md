MECHANISM: Identity-initialized squeeze-excitation residual gating

HYPOTHESIS: Channel-adaptive residual gating will exceed 9,240 correct predictions by adding lightweight backbone selectivity while preserving the strongest verified schedule, augmentation, and classifier.

INTENDED_EDIT: Add zero-initialized squeeze-excitation gates to both residual depthwise blocks, increasing learned parameters from 245,040 to 248,082 with negligible spatial computation.

EVIDENCE: The 245,040-parameter backbone reached 9,240 correct, while classifier reallocations failed and spatial augmentation repeatedly timed out; a low-cost backbone refinement tests an orthogonal source of improvement without changing the verified training procedure.

<<<<<<< SEARCH
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.depthwise(inputs)))
        hidden = self.norm2(self.pointwise(hidden))
        return F.gelu(inputs + hidden)
=======
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

        bottleneck = max(channels // 8, 4)
        self.channel_reduce = nn.Linear(channels, bottleneck)
        self.channel_expand = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.channel_expand.weight)
        nn.init.zeros_(self.channel_expand.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.depthwise(inputs)))
        hidden = self.norm2(self.pointwise(hidden))
        context = hidden.mean(dim=(-2, -1))
        scale = 1.0 + torch.tanh(
            self.channel_expand(F.gelu(self.channel_reduce(context)))
        )
        hidden = hidden * scale[:, :, None, None]
        return F.gelu(inputs + hidden)
>>>>>>> REPLACE