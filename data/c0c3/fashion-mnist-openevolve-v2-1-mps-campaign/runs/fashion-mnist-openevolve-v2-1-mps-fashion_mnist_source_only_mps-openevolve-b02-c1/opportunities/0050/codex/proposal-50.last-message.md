MECHANISM: Relational spatial tokens with class-query attention

HYPOTHESIS: Replacing the fixed flattened MLP with a lightweight relational token head will exceed 9,286 correct predictions because class-specific queries can select contextualized image parts instead of relying on one static mixing of absolute spatial cells.

INTENDED_EDIT: Preserve the proven convolutional backbone, augmentation, optimizer, TTA, and tail averaging, but contextualize the final nine spatial features with self-attention and compute each class logit from a learned query attending to those relational features.

EVIDENCE: Parameter-averaging adjustments plateaued at 9,286 correct and stronger averaging fell to 9,276, while prior backbone and fixed-pooling changes hurt. This challenges the remaining load-bearing assumption—the flattened first-order prediction head—without disturbing the backbone, and uses only nine spatial tokens to avoid the covariance branch’s expensive computation.

<<<<<<< SEARCH
BATCH_SIZE = 48
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 48
GRAD_CLIP_NORM = 1.0


class SpatialRelationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(96)
        self.attention = nn.MultiheadAttention(
            96, num_heads=4, dropout=0.10, batch_first=True
        )
        self.norm2 = nn.LayerNorm(96)
        self.mlp = nn.Sequential(
            nn.Linear(96, 144),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(144, 96),
            nn.Dropout(0.15),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class RelationalClassHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_positions = nn.Parameter(torch.empty(1, 9, 96))
        self.class_tokens = nn.Parameter(torch.empty(1, 10, 96))
        self.spatial_relations = SpatialRelationBlock()
        self.class_norm = nn.LayerNorm(96)
        self.memory_norm = nn.LayerNorm(96)
        self.class_attention = nn.MultiheadAttention(
            96, num_heads=4, dropout=0.10, batch_first=True
        )
        self.output_norm = nn.LayerNorm(96)
        self.class_mlp = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 96),
            nn.Dropout(0.15),
        )
        self.class_readout = nn.Parameter(torch.empty(10, 96))
        self.class_bias = nn.Parameter(torch.zeros(10))

        nn.init.normal_(self.spatial_positions, std=0.02)
        nn.init.normal_(self.class_tokens, std=0.02)
        nn.init.normal_(self.class_readout, std=96 ** -0.5)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        spatial = feature_maps.flatten(2).transpose(1, 2)
        spatial = self.spatial_relations(spatial + self.spatial_positions)

        class_states = self.class_tokens.expand(feature_maps.shape[0], -1, -1)
        attended = self.class_attention(
            self.class_norm(class_states),
            self.memory_norm(spatial),
            self.memory_norm(spatial),
            need_weights=False,
        )[0]
        class_states = class_states + attended
        class_states = class_states + self.class_mlp(
            self.output_norm(class_states)
        )
        return (
            class_states * self.class_readout.unsqueeze(0)
        ).sum(dim=-1) + self.class_bias


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
=======
        self.classifier = RelationalClassHead()
>>>>>>> REPLACE