MECHANISM: Faster curriculum-synchronized all-tensor EMA

HYPOTHESIS: Increasing the residual model’s terminal EMA rate from 0.015 to 0.02 will exceed 9,201 correct predictions by further reducing parameter and BatchNorm-state lag during inference-aligned training.

INTENDED_EDIT: Restore the verified residual topology and BatchNorm-buffer averaging, then use EMA rate 0.005 before the 19/32 curriculum transition and 0.02 afterward.

EVIDENCE: On the same residual curriculum, increasing terminal EMA from 0.005 to 0.01 improved correct predictions from 9,193 to 9,200, and increasing it to 0.015 improved them to 9,201 while also lowering cross-entropy.

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


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
=======
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
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
            ema_rate = 0.02 if step * 32 >= total_steps * 19 else 0.005
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE