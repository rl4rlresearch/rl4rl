MECHANISM: Identity-initialized low-resolution depthwise residual refinement

HYPOTHESIS: Adding a lightweight residual block at 3×3 resolution will exceed 9,286 correct predictions while remaining below 250,000 parameters and avoiding the runtime cost that caused earlier deeper residual designs to time out.

INTENDED_EDIT: Add a zero-initialized 96-channel depthwise-separable residual block after the final pooling layer, increasing the model from 232,682 to 243,146 learned parameters.

EVIDENCE: Temperature calibration has plateaued at 9,286 correct, so further improvement requires changing representation learning. The earlier residual design timed out; concentrating one refinement block at 3×3 resolution tests the same promising feature-reuse mechanism with negligible spatial computation.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 1.0


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(inputs)
        residual = F.gelu(self.norm1(residual))
        residual = self.norm2(self.pointwise(residual))
        return inputs + residual


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
            DepthwiseResidual(96),
        )
>>>>>>> REPLACE