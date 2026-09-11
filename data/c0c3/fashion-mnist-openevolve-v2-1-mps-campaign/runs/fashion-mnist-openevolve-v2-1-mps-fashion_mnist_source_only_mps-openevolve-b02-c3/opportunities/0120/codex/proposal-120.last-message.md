MECHANISM: Content-adaptive nonlocal spatial refinement

HYPOTHESIS: Replacing global channel gating with spatial self-attention will exceed 9,348 correct predictions by routing shape evidence between related image regions while preserving the 7×7 spatial grid.

INTENDED_EDIT: Replace the refinement block’s globally pooled channel gate with learned query-key attention over all 49 feature positions and a channel-wise residual scale; parameter count decreases by seven.

EVIDENCE: Translation TTA raised correctness from 9,344 to 9,348, exposing residual shift sensitivity, while global pooling fell to 9,085 because it discarded spatial organization. This challenges the old assumption that global context should only rescale channels; the new mechanism exchanges content between positions without collapsing layout.

<<<<<<< SEARCH
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
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
=======
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)

        attention_channels = max(channels // 9, 1)
        self.query = nn.Conv2d(
            channels,
            attention_channels,
            kernel_size=1,
            bias=False,
        )
        self.key = nn.Conv2d(
            channels,
            attention_channels,
            kernel_size=1,
            bias=False,
        )
        self.attention_scale = attention_channels ** -0.5

        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.residual_scale = nn.Parameter(torch.ones(channels))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        batch, channels, height, width = refined.shape

        query = self.query(refined).flatten(2).transpose(1, 2)
        key = self.key(refined).flatten(2)
        attention = F.softmax(
            torch.bmm(query, key) * self.attention_scale,
            dim=-1,
        )
        values = refined.flatten(2).transpose(1, 2)
        context = torch.bmm(attention, values)
        context = context.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )

        refined = self.pointwise_norm(
            self.pointwise(refined + context)
        )
        scale = self.residual_scale.view(1, channels, 1, 1)
        return F.gelu(images + scale * refined)
>>>>>>> REPLACE