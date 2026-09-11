MECHANISM: LayerNorm scale packed into an attention-row null direction

HYPOTHESIS: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted-bias design while storing one final LayerNorm scale in the mean of a QKV row, a direction functionally inactive on mean-zero pre-attention inputs.

INTENDED_EDIT: Adopt fourteen-bin far-distance sharing and the verified tilted MLP bias, then remove one explicit final-normalization scale and derive it from an otherwise inactive QKV row mean without changing dense-weight initialization.

EVIDENCE: The tilted fourteen-bin model achieved 99.31% at 1,270 parameters. Earlier gauge packing altered and centered three carrier rows and achieved 89.14%; this tests a single carrier while preserving the original dense QKV layout and initialization.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one scale packed into an external null direction."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        packed_scale = 1.0 + carrier.reshape(1)
        weight = torch.cat((self.weight, packed_scale, self.weight.new_ones(1)))
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(11))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 11),
=======
                self.far_rel_bias.expand(self.n_head, 14),
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with a tilted contrast between hidden-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_contrast = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = torch.cat(
            (self.bias_contrast, (-self.bias_contrast / 3).expand(2))
        )
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.full_weight())
=======
        carrier = self.blocks[0].attn.qkv.weight[0].mean()
        x = self.ln_f(x, carrier)
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE