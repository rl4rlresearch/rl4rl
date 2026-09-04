MECHANISM: BatchNorm-consistent tail weight averaging

HYPOTHESIS: Averaging BatchNorm running statistics alongside the final 10% of model parameters will exceed 9,282 correct predictions by eliminating the normalization-state mismatch in the best tail-averaged model.

INTENDED_EDIT: Restore the verified spatial-refinement architecture with cardinal augmentation and tail averaging, then average and install all floating-point model buffers together with the learned parameters.

EVIDENCE: Reference Design 2 achieved the best result at 9,282 correct using final-10% parameter averaging, but retained terminal BatchNorm statistics; synchronizing those statistics directly strengthens the only optimization change that has surpassed 9,262.

<<<<<<< SEARCH
class PositionalAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = 3
        self.head_dim = 8
        self.inner_channels = self.heads * self.head_dim
        self.qkv = nn.Conv2d(
            channels,
            3 * self.inner_channels,
            kernel_size=1,
            bias=False,
        )
        self.position = nn.Parameter(
            torch.empty(1, self.inner_channels, 7, 7)
        )
        nn.init.normal_(self.position, std=0.02)
        self.project = nn.Sequential(
            nn.Conv2d(
                self.inner_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        locations = height * width
        queries, keys, values = self.qkv(features).chunk(3, dim=1)
        queries = queries + self.position
        keys = keys + self.position

        queries = queries.reshape(
            batch, self.heads, self.head_dim, locations
        ).transpose(-2, -1)
        keys = keys.reshape(
            batch, self.heads, self.head_dim, locations
        )
        values = values.reshape(
            batch, self.heads, self.head_dim, locations
        ).transpose(-2, -1)

        attention = torch.softmax(
            torch.matmul(queries, keys) * (self.head_dim ** -0.5),
            dim=-1,
        )
        mixed = torch.matmul(attention, values)
        mixed = mixed.transpose(-2, -1).reshape(
            batch, self.inner_channels, height, width
        )
        return F.gelu(features + self.project(mixed))
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.late_attention = PositionalAttention(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 60),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(60, 10),
        )
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.late_context(features)
        features = self.late_attention(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
=======
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
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
    optimizer.tail_source_buffers = [
        buffer
        for buffer in model.buffers()
        if buffer.is_floating_point()
    ]
    optimizer.tail_average_buffers = [
        buffer.detach().clone()
        for buffer in optimizer.tail_source_buffers
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
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
            for average, buffer in zip(
                optimizer.tail_average_buffers,
                optimizer.tail_source_buffers,
            ):
                average.lerp_(buffer, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
                for buffer, average in zip(
                    optimizer.tail_source_buffers,
                    optimizer.tail_average_buffers,
                ):
                    buffer.copy_(average)
>>>>>>> REPLACE