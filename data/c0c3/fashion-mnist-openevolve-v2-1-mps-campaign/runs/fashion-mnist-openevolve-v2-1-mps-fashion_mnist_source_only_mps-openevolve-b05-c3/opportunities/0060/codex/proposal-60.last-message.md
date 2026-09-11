MECHANISM: Stagewise residual feature refinement

HYPOTHESIS: On the verified 19/32 curriculum, residual refinement within each convolutional stage will exceed 9,172 correct predictions by preserving useful early-stage features and improving gradient flow during the fixed two-pass exposure budget.

INTENDED_EDIT: Replace the plain six-layer convolutional chain with three parameter-neutral residual stages while retaining the validated positional head and TTA; restore default AdamW and the best 19/32 curriculum.

EVIDENCE: The positional 48-unit head reached 9,172 correct, while widening it fell to 9,164 and attention pooling reportedly fell to 9,103. This suggests preserving the validated prediction head and challenging the load-bearing assumption that each second convolution should completely overwrite its stage representation.

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1.5e-4,
    )
=======
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        weight_decay=1.5e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE