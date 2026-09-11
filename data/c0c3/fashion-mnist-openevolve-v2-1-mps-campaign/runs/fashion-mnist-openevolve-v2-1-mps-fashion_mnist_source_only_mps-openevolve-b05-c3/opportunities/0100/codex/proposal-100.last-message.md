MECHANISM: Identity-initialized hierarchical channel attention

HYPOTHESIS: Adding image-conditioned channel recalibration after each residual stage will exceed 9,206 correct predictions by using global context to reshape intermediate spatial features, while identity initialization preserves the qualified residual model’s starting behavior.

INTENDED_EDIT: Restore the strongest residual architecture, split parameter/BatchNorm EMA, and 1.05× sharpening; add lightweight squeeze-excitation modules within the 250,000-parameter limit.

EVIDENCE: The residual design reached 9,206 correct versus the current plain network’s 9,168. The late global-moment branch fell to 9,201, suggesting global statistics are more informative when they modulate hierarchical feature extraction instead of acting as an additive final classifier.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
        for index, (in_channels, out_channels) in enumerate(channels):
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(4, channels // 8)
        self.reduce = nn.Linear(channels, hidden_channels)
        self.expand = nn.Linear(hidden_channels, channels)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = features.mean(dim=(2, 3))
        gates = self.expand(F.silu(self.reduce(context)))
        return features * (2.0 * torch.sigmoid(gates))[:, :, None, None]


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.ModuleList(
            (
                ResidualStage(1, 32),
                ResidualStage(32, 64),
                ResidualStage(64, 96),
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
        self.attention = nn.ModuleList(
            (
                ChannelAttention(32),
                ChannelAttention(64),
                ChannelAttention(96),
            )
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        for stage, attention in zip(self.features, self.attention):
            features = attention(stage(features))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
=======
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    optimizer._ema_tensors = [
        tensor
        for tensor in parameters + buffers
        if tensor.is_floating_point()
    ]
    buffer_ids = {id(tensor) for tensor in buffers}
    optimizer._ema_is_buffer = [
        id(tensor) in buffer_ids for tensor in optimizer._ema_tensors
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
        else:
            terminal_phase = step * 32 >= total_steps * 19
            for average, tensor, is_buffer in zip(
                optimizer._ema_values,
                optimizer._ema_tensors,
                optimizer._ema_is_buffer,
            ):
                if terminal_phase:
                    ema_rate = 0.0325 if is_buffer else 0.015
                else:
                    ema_rate = 0.005
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE