MECHANISM: Factorized coordinate-attention refinement

HYPOTHESIS: Image-conditioned row/column gating will exceed 9,285 correct predictions by preserving garment-part arrangement before global pooling while reducing refinement computation enough to complete verification.

INTENDED_EDIT: Replace the assumption that fixed global mean/max statistics alone capture spatial structure with factorized axial attention that modulates each channel by learned row and column context; slightly narrow the classifier to remain below 250,000 parameters and restore the stronger default BatchNorm behavior.

EVIDENCE: Global-pooling variants plateaued at 9,285 correct while averaging changes moved only 2–4 predictions; spatial refinement previously improved 9,202 to 9,209, but the axial-profile head timed out. Applying compressed axial context before pooling tests the spatial mechanism while replacing the expensive 96×96 pointwise refinement.

<<<<<<< SEARCH
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))
=======
class CoordinateAttentionRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck = 32
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.GELU(),
        )
        self.row_gate = nn.Conv2d(
            bottleneck, channels, kernel_size=1, bias=True
        )
        self.column_gate = nn.Conv2d(
            bottleneck, channels, kernel_size=1, bias=True
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.gelu(features + self.local(features))
        row_context = features.mean(dim=3, keepdim=True)
        column_context = features.mean(dim=2, keepdim=True).transpose(2, 3)
        height = row_context.shape[2]

        context = self.context(
            torch.cat((row_context, column_context), dim=2)
        )
        row_context, column_context = torch.split(
            context,
            (height, context.shape[2] - height),
            dim=2,
        )
        row_attention = torch.sigmoid(self.row_gate(row_context))
        column_attention = torch.sigmoid(
            self.column_gate(column_context).transpose(2, 3)
        )
        return features * (1.0 + row_attention * column_attention)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = CoordinateAttentionRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 60),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(60, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    model = ImageClassifier()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.04
    return model
=======
def build_model() -> nn.Module:
    return ImageClassifier()
>>>>>>> REPLACE