MECHANISM: Multi-scale relative-position self-attention refinement

HYPOTHESIS: Replacing fixed depthwise spatial refinement and global channel gating with content-adaptive token interactions will exceed 9,322 correct predictions while preserving the load-bearing 7×7 spatial layout.

INTENDED_EDIT: Replace `SpatialRefinement` with four-head self-attention using learned relative-position biases initialized at multiple locality scales, and reduce the dense bottleneck from 56 to 53 units to remain at 249,429 learned parameters.

EVIDENCE: Global pooling fell to 9,085 correct, proving spatial layout is essential, while repeated global-gate variants peaked at 9,322 and usually regressed. This retains every spatial location but challenges the shared assumption that interactions between them should be fixed convolutions modulated only by a global descriptor.

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
        pooled = F.adaptive_avg_pool2d(refined, output_size=1)
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
        return F.gelu(images + gate * refined)
=======
class SpatialAttentionRefinement(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)

        relative_axis = torch.arange(-6, 7)
        delta_y, delta_x = torch.meshgrid(
            relative_axis,
            relative_axis,
            indexing="ij",
        )
        distance = delta_y.abs() + delta_x.abs()
        locality = torch.linspace(0.0, 0.45, heads).view(heads, 1, 1)
        self.relative_bias = nn.Parameter(
            -locality * distance.unsqueeze(0)
        )

        grid = torch.arange(7)
        coordinates = torch.stack(
            torch.meshgrid(grid, grid, indexing="ij")
        ).flatten(1)
        relative_offsets = (
            coordinates[:, :, None] - coordinates[:, None, :] + 6
        )
        self.register_buffer(
            "relative_offsets",
            relative_offsets,
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        token_count = height * width
        tokens = images.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)

        qkv = self.qkv(normalized).reshape(
            batch,
            token_count,
            3,
            self.heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_logits = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * self.scale
        position_bias = self.relative_bias[
            :,
            self.relative_offsets[0],
            self.relative_offsets[1],
        ]
        attention = F.softmax(
            attention_logits + position_bias.unsqueeze(0),
            dim=-1,
        )
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).reshape(
            batch,
            token_count,
            channels,
        )
        tokens = F.gelu(tokens + self.projection(context))
        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            SpatialRefinement(64),
=======
            SpatialAttentionRefinement(64),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
=======
            nn.Linear(64 * 7 * 7, 53),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(53, 10),
>>>>>>> REPLACE