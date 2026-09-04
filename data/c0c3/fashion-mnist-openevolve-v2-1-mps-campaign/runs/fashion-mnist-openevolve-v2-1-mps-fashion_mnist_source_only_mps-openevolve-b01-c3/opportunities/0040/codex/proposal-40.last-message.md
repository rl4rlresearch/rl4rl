MECHANISM: Content-adaptive nonlocal spatial relation block

HYPOTHESIS: Replacing fixed-kernel late refinement with lightweight four-head self-attention will exceed 9,285 correct predictions by learning image-dependent relationships between distant garment parts before global pooling.

INTENDED_EDIT: Replace the 7×7 depthwise refinement block with relative-position-aware spatial self-attention, while restoring cardinal-shift augmentation and the strongest verified final-10% linear recency-weighted averaging procedure.

EVIDENCE: Local refinement improved correctness from 9,202 to 9,209, showing value in late spatial interaction, while static attention pooling reached only 9,257; Reference Design 3 reached 9,285 with linear tail averaging. This challenges the load-bearing assumption that fixed local kernels provide sufficient spatial reasoning, using dynamic all-pairs feature interaction rather than another pooling modification.

<<<<<<< SEARCH
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))
=======
class SpatialRelation(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 4,
        head_dim: int = 6,
        feature_size: int = 7,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.attention_channels = heads * head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(
            channels,
            3 * self.attention_channels,
            kernel_size=1,
            bias=False,
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                self.attention_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        relative_size = 2 * feature_size - 1
        self.relative_bias = nn.Parameter(
            torch.zeros(heads, relative_size * relative_size)
        )
        coordinates = torch.arange(feature_size)
        grid = torch.stack(
            torch.meshgrid(coordinates, coordinates, indexing="ij")
        ).flatten(1)
        relative = grid[:, :, None] - grid[:, None, :]
        relative = relative + feature_size - 1
        relative_index = relative[0] * relative_size + relative[1]
        self.register_buffer(
            "relative_index",
            relative_index,
            persistent=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        tokens = height * width
        qkv = self.qkv(self.norm(features)).reshape(
            batch,
            3,
            self.heads,
            self.head_dim,
            tokens,
        )
        qkv = qkv.permute(1, 0, 2, 4, 3)
        queries, keys, values = qkv.unbind(0)

        attention = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        ) * self.scale
        attention = attention + self.relative_bias[:, self.relative_index]
        attention = attention.softmax(dim=-1)

        mixed = torch.matmul(attention, values)
        mixed = mixed.transpose(-2, -1).reshape(
            batch,
            self.attention_channels,
            height,
            width,
        )
        return F.gelu(features + self.output(mixed))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.late_refinement = SpatialRefinement(96)
=======
        self.late_refinement = SpatialRelation(96)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=2e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        weight_decay=2e-4,
    )
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")

    row_offsets = torch.randint(0, 5, (batch, 1), device=images.device)
    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    col_offsets = torch.randint(0, 5, (batch, 1), device=images.device)
    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
    col_index = cols[:, None, None, :].expand(batch, channels, height, width)
    images = images.gather(3, col_index)

    flip_mask = torch.rand(batch, device=images.device) < 0.5
=======
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)

    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
    col_index = cols[:, None, None, :].expand(batch, channels, height, width)
    images = images.gather(3, col_index)

    flip_mask = torch.rand(batch, device=images.device) < 0.5
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
>>>>>>> REPLACE