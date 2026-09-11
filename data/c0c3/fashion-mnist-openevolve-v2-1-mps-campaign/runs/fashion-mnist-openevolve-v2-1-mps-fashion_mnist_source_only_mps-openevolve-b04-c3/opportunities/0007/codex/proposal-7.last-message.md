MECHANISM: Compute-light residual classifier refinement

HYPOTHESIS: Adding a near-capacity residual MLP to the qualified batch-64 model will exceed 9,229 correct predictions while completing within the time limit because it adds feature refinement with negligible spatial computation.

INTENDED_EDIT: Use batch size 64 and spend 4,864 remaining parameters on a pre-normalized 38→63→38 residual MLP, bringing the model to 249,904 parameters.

EVIDENCE: Reference Design 2 achieved 9,229 correct at batch size 64, while adding a residual block at 14×14 timed out; placing residual capacity in the compact classifier tests additional depth without that high-resolution runtime cost.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.gelu(inputs + hidden)


class ImageClassifier(nn.Module):
=======
        return F.gelu(inputs + hidden)


class ResidualMLPBlock(nn.Module):
    def __init__(self, channels: int, expansion: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, expansion, bias=False)
        self.project = nn.Linear(expansion, channels, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.expand(self.norm(inputs)))
        return inputs + self.project(hidden)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
=======
            nn.LayerNorm(38),
            nn.GELU(),
            ResidualMLPBlock(38, 63),
            nn.Dropout(p=0.1),
>>>>>>> REPLACE