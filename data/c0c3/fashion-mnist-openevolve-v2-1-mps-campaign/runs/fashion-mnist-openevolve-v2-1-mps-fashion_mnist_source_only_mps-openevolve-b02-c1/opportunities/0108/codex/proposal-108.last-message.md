MECHANISM: Identity-initialized efficient channel attention

HYPOTHESIS: A three-parameter input-dependent channel gate after the terminal pool will achieve at least 9,311 validation-correct predictions while retaining the verified runtime profile and 249,601-parameter architecture.

INTENDED_EDIT: Insert an identity-initialized global-context channel gate after the existing terminal max-pool, adding only three learned parameters and leaving training, TTA, and the normalized 147-unit head unchanged.

EVIDENCE: The normalized dense head produced the best result at 9,310 correct, while a 10,464-parameter spatial/channel residual block timed out; this tests adaptive channel refinement with negligible computation and without disturbing the successful pooling or classifier paths.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 1.0


class EfficientChannelAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        nn.init.zeros_(self.context.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(2, 3)).unsqueeze(1)
        weights = 2.0 * torch.sigmoid(self.context(pooled))
        return features * weights.squeeze(1).unsqueeze(-1).unsqueeze(-1)


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
            EfficientChannelAttention(),
        )
>>>>>>> REPLACE