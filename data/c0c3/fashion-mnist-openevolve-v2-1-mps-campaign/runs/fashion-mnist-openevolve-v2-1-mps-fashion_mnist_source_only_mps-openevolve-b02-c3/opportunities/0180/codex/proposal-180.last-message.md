MECHANISM: Position-aware nonlocal spatial attention

HYPOTHESIS: Adding content-dependent all-to-all interactions across the preserved 7×7 feature grid will exceed 9,348 correct predictions by modeling relationships between distant garment regions that fixed local refinement cannot capture.

INTENDED_EDIT: Retain the successful local refinement and full-grid classifier, insert a residual four-head attention mixer with learned spatial positions, and reduce the dense bottleneck from 56 to 49 units to keep 247,820 learned parameters.

EVIDENCE: Global pooling fell to 9,085 because it discarded spatial layout, while parallel local refinements reached only 9,321; this motivates a mechanism that preserves every spatial position but adds genuinely nonlocal, image-dependent aggregation.

<<<<<<< SEARCH
        return F.gelu(images + gate * refined)


class ImageClassifier(nn.Module):
=======
        return F.gelu(images + gate * refined)


class NonlocalSpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.empty(1, 7 * 7, channels))
        nn.init.normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads=4,
            batch_first=True,
        )
        self.residual_scale = nn.Parameter(
            torch.full((channels,), 0.1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        tokens = images.flatten(2).transpose(1, 2)
        positioned_tokens = self.norm(tokens + self.position)
        attended_tokens = self.attention(
            positioned_tokens,
            positioned_tokens,
            positioned_tokens,
            need_weights=False,
        )[0]
        tokens = tokens + self.residual_scale * attended_tokens
        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            SpatialRefinement(64),
=======
            SpatialRefinement(64),
            NonlocalSpatialAttention(64),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
=======
            nn.Linear(64 * 7 * 7, 49),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(49, 10),
>>>>>>> REPLACE