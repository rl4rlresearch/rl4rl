MECHANISM: LayerNorm-nullspace weight gauge fixing

HYPOTHESIS: Removing one `fc1` input-weight coordinate from the verified distributed-anchor design will yield 1,609 parameters and at least 99% accuracy, because bias-free LayerNorm outputs have zero mean and the omitted coefficient is exactly absorbable by the other coefficients in that output row; gauge-adjusted initialization preserves the original model’s initial function and RNG stream.

INTENDED_EDIT: Anchor one scale coordinate in each pre-sublayer LayerNorm, then replace `fc1` with a linear layer that learns all but one weight, synthesizes the omitted weight as zero, and initializes the retained row coefficients to an equivalent gauge.

EVIDENCE: The distributed LayerNorm-anchor design achieved 99.97% with 1,610 parameters. Both prior 1,609-parameter attempts altered sensitive additive pathways and failed, motivating a different one-parameter reduction based on the exact zero-mean nullspace immediately downstream of `ln2`.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)


class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(base.weight.new_empty(out_features * in_features - 1))
        self.bias = nn.Parameter(base.bias.new_empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.in_features - 1
        weight = torch.cat(
            (self.weight[:anchor], self.weight.new_zeros(1), self.weight[anchor:])
        ).view(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Pre-sublayer normalization offsets are absorbed by downstream biases.
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(module.out_features, module.in_features)
                nn.init.normal_(full, mean=0.0, std=0.02)

                # For normalized z, sum(z) is zero. Subtracting the omitted
                # coefficient from the other row coefficients preserves row 0.
                full[0, :-1].sub_(full[0, -1].clone())
                flat = full.flatten()
                anchor = module.in_features - 1
                module.weight.copy_(torch.cat((flat[:anchor], flat[anchor + 1 :])))
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE