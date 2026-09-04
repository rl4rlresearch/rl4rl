MECHANISM: Zero-initialized residual classifier refinement

HYPOTHESIS: Adding a zero-initialized nonlinear residual transformation to the 48-dimensional classifier representation, atop the verified 0.04→0 smoothing schedule, will exceed 9,240 correct predictions without destabilizing the successful initial optimization path.

INTENDED_EDIT: Insert a lightweight residual MLP block in the classifier, adopt the best verified label-smoothing schedule, and use the verified 1.296352 evaluation-logit calibration; total learned parameters become 244,994.

EVIDENCE: The 0.04→0 smoothing design retained 9,240 correct and achieved the best available cross-entropy, while spatial gating reduced correctness and attention exceeded the time limit. A zero-initialized residual head adds inexpensive global nonlinear capacity while initially preserving the proven classifier exactly.

<<<<<<< SEARCH
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ResidualClassifierBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + F.gelu(self.projection(self.norm(inputs)))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
=======
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            ResidualClassifierBlock(48),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.30 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE