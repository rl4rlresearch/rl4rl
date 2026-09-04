MECHANISM: Terminal spatial-relation attention

HYPOTHESIS: Content-dependent attention among the nine terminal spatial tokens will exceed 9,311 correct predictions by resolving shape-part relationships that the fixed flattened bottleneck cannot model.

INTENDED_EDIT: Reallocate classifier capacity from the 147-unit flattening layer to four-head self-attention over the 3×3 feature grid, followed by a smaller positional classifier; retain the established convolutional trunk, training procedure, and TTA.

EVIDENCE: Full-resolution token interaction timed out, suggesting interaction cost was load-bearing rather than disproving relational modeling; applying attention only after reduction to nine tokens tests the mechanism cheaply. Translation-aware aggregation achieved the best 9,311 correct, further indicating that spatial organization affects borderline predictions.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class SpatialRelationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.empty(1, 9, 96))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(96)
        self.attention = nn.MultiheadAttention(
            embed_dim=96,
            num_heads=4,
            dropout=0.05,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 96, 103),
            nn.BatchNorm1d(103),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(103, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.position
        normalized = self.norm(tokens)
        relations, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        return self.classifier(tokens + relations)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
=======
        self.classifier = SpatialRelationHead()
>>>>>>> REPLACE