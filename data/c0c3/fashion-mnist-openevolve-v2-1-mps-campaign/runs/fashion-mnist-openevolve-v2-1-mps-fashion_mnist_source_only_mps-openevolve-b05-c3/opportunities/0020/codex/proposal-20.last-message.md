MECHANISM: Multi-scale global channel recalibration

HYPOTHESIS: Adding identity-initialized, image-conditioned channel gates at each convolutional stage will exceed 9,167 correct predictions by injecting global shape context while preserving the validated spatially indexed classifier and convolution widths.

INTENDED_EDIT: Use the best validated 39/64 augmentation boundary and add lightweight squeeze-and-excitation-style channel recalibration before each pooling operation, increasing the model to 249,618 learned parameters.

EVIDENCE: Spatial attention pooling fell to 9,103 correct, indicating that replacing the position-sensitive head discards useful structure; this alternative retains that head and all validated compute-heavy layers while testing global contextual feature modulation with negligible additional spatial computation.

<<<<<<< SEARCH
BASE_LR = 3.0e-3


class ImageClassifier(nn.Module):
=======
BASE_LR = 3.0e-3


class ChannelRecalibration(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(4, channels // 8)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(features, 1)
        gates = 2.0 * torch.sigmoid(
            self.expand(F.silu(self.reduce(context)))
        )
        return features * gates


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
=======
            if index in (1, 3, 5):
                layers.extend(
                    (
                        ChannelRecalibration(out_channels),
                        nn.MaxPool2d(2),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE