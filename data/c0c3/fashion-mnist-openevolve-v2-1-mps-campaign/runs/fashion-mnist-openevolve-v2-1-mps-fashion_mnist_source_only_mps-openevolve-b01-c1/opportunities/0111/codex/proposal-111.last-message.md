MECHANISM: Anti-aliased max downsampling

HYPOTHESIS: Preserving local maxima while low-pass filtering before each stride-2 reduction will exceed 9,257 correct predictions by reducing the shift aliasing exposed by crop-weighted validation without sacrificing salient garment features.

INTENDED_EDIT: Add a parameter-free max-then-binomial-blur pooling layer and replace both standard max-pooling stages with it.

EVIDENCE: Center-weighted crop aggregation improved correctness, indicating downsampling-phase sensitivity, while learned max/average mixing fell to 9,246; anti-aliased max pooling targets that sensitivity while retaining the max operation that the mixture weakened.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0


class BlurMaxPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer(
            "kernel",
            kernel_2d.expand(channels, 1, 3, 3).contiguous(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.max_pool2d(features, kernel_size=2, stride=1)
        features = F.pad(features, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(
            features,
            self.kernel,
            stride=2,
            groups=features.shape[1],
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(16),
            nn.GELU(),
            BlurMaxPool2d(16),
            nn.Conv2d(16, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            BlurMaxPool2d(40),
>>>>>>> REPLACE