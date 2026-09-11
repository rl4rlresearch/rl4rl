MECHANISM: Zero-initialized depthwise spatial residual refinement

HYPOTHESIS: Adding a trainable per-channel 3×3 residual refinement over the final feature map will exceed 9,166 correct predictions by learning local spatial context while preserving the successful representation at initialization and adding negligible runtime.

INTENDED_EDIT: Add a zero-initialized depthwise residual layer after the final convolution and reduce the dense hidden width from 160 to 159, keeping the model under the parameter ceiling at 249,970 parameters.

EVIDENCE: Learned zero-DC detail filters improved correctness to 9,166, supporting adaptive spatial filtering; prior heavier spatial-context attempts timed out, motivating a depthwise residual refinement with minimal computation and an identity initialization.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class DepthwiseSpatialRefinement(nn.Module):
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
        self.normalization = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.normalization(self.depthwise(inputs))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
=======
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            DepthwiseSpatialRefinement(96),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
=======
            nn.Linear(96 * 3 * 3, 159),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(159, 10),
>>>>>>> REPLACE