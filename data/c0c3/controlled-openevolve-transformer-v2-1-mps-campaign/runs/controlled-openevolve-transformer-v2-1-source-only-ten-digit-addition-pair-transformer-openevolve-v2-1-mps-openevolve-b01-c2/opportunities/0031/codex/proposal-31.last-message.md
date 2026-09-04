MECHANISM: Affine-free attention normalization with exact QKV row gauges

HYPOTHESIS: Removing `ln1`’s redundant learned scale and gauge-fixing every QKV row will produce a 1,560-parameter model with at least 99% accuracy.

INTENDED_EDIT: Make both pre-sublayer normalizations affine-free, extend `fc1` gauge fixing to all rows, and omit one LayerNorm-null coefficient from every learned QKV projection row.

EVIDENCE: The analogous affine-free `ln2` plus all-row `fc1` gauge achieved 99.96% at 1,591 parameters; applying the same exact normalized-input nullspace argument to `ln1` and QKV is the closest supported reduction.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model

        # Match the constructor RNG consumption of the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * (self.in_features - 1)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        # Key bias cancels in softmax; value bias is absorbed by proj.bias.
        self.bias = nn.Parameter(base.bias.new_empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact = self.weight.view(self.out_features, self.in_features - 1)
        weight = torch.cat(
            (compact, compact.new_zeros(self.out_features, 1)), dim=1
        )
        fused_bias = torch.cat(
            (self.bias, self.bias.new_zeros(2 * self.in_features))
        )
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the baseline projection construction and RNG stream, while
        # retaining only query bias. Key bias cancels in the softmax, and value
        # bias is a constant offset absorbed by the output projection bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        self.qkv = LayerNormGaugedQKV(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
=======
        qkv = self.qkv(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8, 9)
=======
        self.gauged_rows = tuple(range(out_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
=======
        # Each following learned projection can absorb LayerNorm feature scales.
        # Affine-free normalization also exposes an exact common-row nullspace.
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
=======
        if isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # Affine-free LayerNorm outputs sum to zero, so subtracting
                # each row's final coefficient preserves the initial function.
                full[:, :-1].sub_(full[:, -1:])
                module.weight.copy_(full[:, :-1].flatten())
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
>>>>>>> REPLACE