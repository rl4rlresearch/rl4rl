MECHANISM: Weight-decay-free output-bias channel

HYPOTHESIS: Removing a fourth final-LayerNorm bias direction while exempting the tied token embedding from weight decay will achieve at least 99% accuracy with 1,582 parameters, because the token-row means that absorb omitted output biases are input-side invisible and should not be penalized toward zero.

INTENDED_EDIT: Reduce the learned final-LayerNorm bias basis from five to four coordinates and use a zero-weight-decay optimizer group for the tied token embedding while retaining configured weight decay elsewhere.

EVIDENCE: The four-direction quotient narrowly missed at 98.28%, whereas the three-direction quotient reached 99.91%; amplifying the fixed bias failed, motivating an optimizer-level change that directly supports the demonstrated token-row-mean absorption channel without changing its activation scale.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with three bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for j in range(normalized_shape - 3):
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with four bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 4)
        for j in range(normalized_shape - 4):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 3))
=======
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    embedding_param = model.token_emb.weight
    decay_params = [
        param for param in model.parameters() if param is not embedding_param
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": [embedding_param], "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
    )
>>>>>>> REPLACE