MECHANISM: Gauge-anchored query projection with adaptive bias sharing

HYPOTHESIS: A 1,402-parameter model will retain at least 99% accuracy because the verified 1,403-parameter design reached 99.98%, while anchoring one zero-bias query coordinate removes only a query-key scaling gauge degree of freedom rather than tying effective attention maps.

INTENDED_EDIT: Apply the verified three-quartet MLP biases and single attention-output bias tie, then replace the 192-parameter QKV matrix with a 191-parameter matrix whose query row 1, column 0 coefficient is fixed at its initialization scale.

EVIDENCE: The 1,403-parameter attention-bias design achieved 99.98%; wholesale query-key sharing failed at 93.49%, motivating a one-scalar gauge anchor that preserves independent query and key coefficients and their effective bilinear attention map.

<<<<<<< SEARCH
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class AnchoredScaleLayerNorm(nn.Module):
=======
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one adaptively shared bias pair."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 3:
            raise ValueError("out_features must be at least three")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.free_bias[:-1], self.free_bias[-1:].expand(2)))
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)


class GaugeAnchoredQKVLinear(nn.Module):
    """Bias-free QKV map with one query-key scaling gauge fixed."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be at least two")
        self.d_model = d_model
        self.anchor_index = d_model
        self.free_weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))
        self.register_buffer(
            "anchor_weight", torch.tensor([0.02]), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_weight = torch.cat(
            (
                self.free_weight[: self.anchor_index],
                self.anchor_weight,
                self.free_weight[self.anchor_index :],
            )
        )
        weight = flat_weight.view(3 * self.d_model, self.d_model)
        return F.linear(x, weight)


class AnchoredScaleLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = GaugeAnchoredQKVLinear(d_model)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose final six outputs form three learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = self.free_bias[-3:].repeat_interleave(2)
        bias = torch.cat((self.free_bias[:-3], shared_biases))
        return F.linear(x, self.weight, bias)
=======
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
        if isinstance(module, GaugeAnchoredQKVLinear):
            nn.init.normal_(module.free_weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE