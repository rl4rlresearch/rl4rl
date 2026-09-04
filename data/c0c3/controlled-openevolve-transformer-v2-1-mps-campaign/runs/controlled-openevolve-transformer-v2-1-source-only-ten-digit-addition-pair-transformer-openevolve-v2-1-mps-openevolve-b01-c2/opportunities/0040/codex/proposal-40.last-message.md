MECHANISM: Single-row QKV LayerNorm-nullspace gauge

HYPOTHESIS: Applying one QKV row gauge to the verified 1,587-parameter design will produce a 1,586-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the affine-free `ln2`, all-row `fc1` gauge, and five anchored `ln1` scales, then omit and reconstruct the final input coefficient of QKV row 0 while preserving its initialized function.

EVIDENCE: Reference Design 2 achieved 99.40% with 1,587 parameters. The affine-free `ln1` experiment failed even without QKV gauges, indicating that removing attention scales was the primary harmful change; retaining three learned scales while testing one exact QKV-row redundancy is the smallest orthogonal reduction.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Preserve the baseline projection construction and RNG stream, while
        # retaining only query bias. Key bias cancels in the softmax, and value
        # bias is a constant offset absorbed by the output projection bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0,)
        self.ungauged_rows = tuple(range(1, self.out_features))

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        gauged = torch.cat(
            (
                self.weight[:row_width].view(1, row_width),
                self.weight.new_zeros(1, 1),
            ),
            dim=1,
        )
        ungauged = self.weight[row_width:].view(
            len(self.ungauged_rows), self.in_features
        )
        weight = torch.cat((gauged, ungauged), dim=0)

        fused_bias = torch.cat(
            (
                self.bias,
                self.bias.new_zeros(self.in_features),
                self.bias.new_zeros(self.in_features),
            )
        )
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Retain query bias and one representative of QKV row 0's normalized-
        # input coefficient gauge.
        self.qkv = LayerNormGaugedQKV(d_model)
        self.proj = nn.Linear(d_model, d_model)
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
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # Affine-free normalization makes every fc1 row gauge exact.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, LayerNormGaugedLinear):
=======
        if isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # All attention scales initialize to one, so subtracting the
                # omitted coefficient preserves the initial row function.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedLinear):
>>>>>>> REPLACE