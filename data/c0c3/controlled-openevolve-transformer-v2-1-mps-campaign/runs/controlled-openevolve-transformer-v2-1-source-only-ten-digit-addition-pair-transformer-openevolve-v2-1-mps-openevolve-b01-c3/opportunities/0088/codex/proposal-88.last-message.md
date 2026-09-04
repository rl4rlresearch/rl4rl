MECHANISM: Single-row LayerNorm-nullspace quotient

HYPOTHESIS: The 1,269-parameter model will achieve at least 99% accuracy because it preserves the verified 1,270-parameter architecture while removing only one functionally inactive row-constant direction from the final value-projection row.

INTENDED_EDIT: Adopt fourteen-bin far-distance sharing and the verified tilted MLP bias, then parameterize one QKV row in an orthonormal mean-zero basis, reducing its eight weights to seven.

EVIDENCE: The tilted fourteen-bin model achieved 99.31% at 1,270 parameters. The 1,237-parameter wholesale LayerNorm-subspace projection disrupted training, motivating a conservative one-row quotient whose effective initialization distribution remains isotropic.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class SingleCenteredRowLinear(nn.Module):
    """Bias-free linear map with one row restricted to the input mean-zero basis."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features - 1, in_features))
        self.centered_weight = nn.Parameter(torch.empty(in_features - 1))
        self.register_buffer(
            "input_basis", mean_zero_basis(in_features), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered_row = self.centered_weight @ self.input_basis.transpose(0, 1)
        weight = torch.cat((self.weight, centered_row.unsqueeze(0)), dim=0)
        return F.linear(x, weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = SingleCenteredRowLinear(d_model, 3 * d_model)
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
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SingleCenteredRowLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.normal_(module.centered_weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE