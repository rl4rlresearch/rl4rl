MECHANISM: Single-coordinate token/position shift-gauge anchoring

HYPOTHESIS: Adding one exactly compensated mean-zero token/position gauge constraint to the qualified 1580-parameter two-anchor model will retain at least 99% accuracy with 1579 parameters.

INTENDED_EDIT: Use the qualified twice-anchored attention projection, then remove one coordinate from the final token embedding row while transferring its initialized contribution into every positional embedding.

EVIDENCE: The two-projection-bias-anchor design achieved 99.97% at 1580 parameters; the one-dimensional global embedding gauge also retained 99.99%, whereas removing all vocabulary-common modes collapsed accuracy, motivating a single compensated gauge reduction rather than a third projection-bias anchor.

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
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and token/position shift gauges fixed."""
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
        shift = last_coords[-1]
        gauged_weight = gauged_weight - basis[:, -1] * shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-1])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-1]
        gauged_weight = gauged_weight - self.basis[:, -1] * shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-1])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 1))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
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


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Preserve the initialized input function after fixing the additional
        # token/position shift gauge.
        with torch.no_grad():
            self.pos_emb.weight[:, -1].add_(
                self.token_emb.initial_position_shift
            )

    @staticmethod
>>>>>>> REPLACE