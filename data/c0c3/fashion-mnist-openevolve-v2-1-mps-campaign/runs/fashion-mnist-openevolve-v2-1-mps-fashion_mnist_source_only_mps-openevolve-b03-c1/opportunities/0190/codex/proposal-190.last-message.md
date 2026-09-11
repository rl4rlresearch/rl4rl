MECHANISM: Full-grid gated bilinear prediction head

HYPOTHESIS: Replacing the additive 160-unit MLP with gated multiplicative interactions over the complete 3×3 feature grid plus a linear evidence path will exceed 9,210 correct predictions while remaining below the parameter ceiling.

INTENDED_EDIT: Preserve the successful spatial convolutional representation and TTA, but compute class logits with a rank-75 GEGLU-style full-grid head and linear skip; estimated parameters decrease from 249,789 to 249,079.

EVIDENCE: The flattened spatial design reached 9,210 correct, whereas global-context and axial-profile alternatives fell to 9,128 and 9,167. This indicates full 2D layout is load-bearing, but does not validate the shared assumption that an ordinary one-hidden-layer additive head is the best way to combine distant spatial evidence.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class GatedSpatialClassifier(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        rank: int,
    ) -> None:
        super().__init__()
        input_features = channels * height * width
        self.input_dropout = nn.Dropout(0.15)
        self.content_gate = nn.Linear(input_features, 2 * rank)
        self.interaction_norm = nn.LayerNorm(rank)
        self.interaction_dropout = nn.Dropout(0.10)
        self.linear_skip = nn.Linear(input_features, 10, bias=False)
        self.output = nn.Linear(rank, 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened = features.flatten(1)
        dropped = self.input_dropout(flattened)
        content, gate = self.content_gate(dropped).chunk(2, dim=1)
        interactions = self.interaction_norm(content * F.gelu(gate))
        return self.linear_skip(dropped) + self.output(
            self.interaction_dropout(interactions)
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.classifier = GatedSpatialClassifier(
            channels=96,
            height=3,
            width=3,
            rank=75,
        )
>>>>>>> REPLACE