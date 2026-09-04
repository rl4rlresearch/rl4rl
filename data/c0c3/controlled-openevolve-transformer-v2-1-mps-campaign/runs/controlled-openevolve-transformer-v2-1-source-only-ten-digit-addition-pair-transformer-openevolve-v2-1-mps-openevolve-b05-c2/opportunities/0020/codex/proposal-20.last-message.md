MECHANISM: Fourth compensated token/position shift-gauge anchor

HYPOTHESIS: Extending the qualified three-coordinate embedding gauge to four coordinates will produce a 1576-parameter model with at least 99% accuracy while preserving its initialized input function and output probabilities.

INTENDED_EDIT: Adopt the qualified twice-anchored attention projection and constrain four coordinates of the final token row, transferring their initialized common shift into every positional embedding.

EVIDENCE: Three compensated gauge anchors achieved 99.8% accuracy at 1577 parameters, while a third attention projection-bias anchor collapsed to 74.33%; this supports extending the successful embedding gauge instead of further restricting attention.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with its global all-ones gauge fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(gauged_weight[-1] @ basis)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(gauged_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ self.last_weight
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and four token/position gauges fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-4:]
        gauged_weight = gauged_weight - basis[:, -4:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-4])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-4:]
        gauged_weight = gauged_weight - self.basis[:, -4:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-4])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 4))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class AnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with one bias coordinate absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
=======
class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two bias coordinates absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = AnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        with torch.no_grad():
            self.pos_emb.weight[:, -4:].add_(
                self.token_emb.initial_position_shift
            )

    @staticmethod
>>>>>>> REPLACE