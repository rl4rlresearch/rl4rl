MECHANISM: Final-LayerNorm common-bias gauge

HYPOTHESIS: Removing the final block MLP output bias’s vocabulary-invariant common-coordinate direction will reduce the model to 1,577 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Replace the final block’s `fc2` with a linear layer whose last bias coordinate is fixed to zero, retaining ordinary linear layers in earlier blocks.

EVIDENCE: The current 1,578-parameter design achieved 99.91%, while a third positional anchor collapsed to 72.81%; this tests an orthogonal exact null direction because the final LayerNorm removes any common-coordinate shift added by the final MLP bias.

<<<<<<< SEARCH
        weight = torch.stack(rows)
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        weight = torch.stack(rows)
        return F.linear(x, weight, self.bias)


class FinalLayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(
            base.weight.new_empty(out_features, in_features)
        )
        self.bias = nn.Parameter(base.bias.new_empty(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Any omitted common-coordinate bias is removed by the final LayerNorm.
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        gauge_output_bias: bool = False,
    ):
        super().__init__()
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        if gauge_output_bias:
            self.fc2 = FinalLayerNormGaugedLinear(d_ff, d_model)
        else:
            self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # fc1 absorbs feature scaling; affine-free normalization also makes
        # every fc1 common-row coefficient an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, gauge_output_bias: bool = False):
        super().__init__()
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # fc1 absorbs feature scaling; affine-free normalization also makes
        # every fc1 common-row coefficient an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(
            cfg.d_model,
            cfg.d_ff,
            cfg.dropout,
            gauge_output_bias=gauge_output_bias,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
=======
        self.blocks = nn.ModuleList(
            [
                Block(cfg, gauge_output_bias=(layer == cfg.n_layer - 1))
                for layer in range(cfg.n_layer)
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, FinalLayerNormGaugedLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE