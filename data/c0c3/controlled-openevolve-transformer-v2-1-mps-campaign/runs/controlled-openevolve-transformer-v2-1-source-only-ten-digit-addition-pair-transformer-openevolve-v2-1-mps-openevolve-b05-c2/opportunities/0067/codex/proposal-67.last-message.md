MECHANISM: Seventh pre-attention LayerNorm scale gauge

HYPOTHESIS: Reproducing the qualified 1513-parameter design and fixing a seventh `ln1` scale will yield a 1512-parameter model with at least 99% accuracy.

INTENDED_EDIT: Complete the token-position embedding gauge, omit `q_bias[4]`, and replace four learned `ln1` scales with one learned scale and seven fixed unit scales.

EVIDENCE: The six-scale design achieved 99.91% accuracy at 1513 parameters, while the preceding five-scale design achieved 99.84%; extending the same successful LayerNorm gauge is the closest supported reduction.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and six token/position shift gauges fixed."""
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
        shift = last_coords[-6:]
        gauged_weight = gauged_weight - basis[:, -6:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-6])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with every token/position common-shift gauge fixed."""
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
        shift = last_coords
        gauged_weight = gauged_weight - basis @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-6:]
        gauged_weight = gauged_weight - self.basis[:, -6:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-6])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 6))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords
        gauged_weight = gauged_weight - self.basis @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.weight_rows.new_zeros(self.embedding_dim)
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias[:self.head_dim],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim:],
            )
        )
        qkv_bias = torch.cat(
            (
                q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuadrupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with four scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class SeptupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with seven scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 7), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = QuadrupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized input function after fixing the additional
        # token/position shift gauge.
        with torch.no_grad():
            self.pos_emb.weight[:, -6:].add_(
                self.token_emb.initial_position_shift
            )
=======
        # Preserve the initialized input function after fixing every
        # token/position shift gauge.
        with torch.no_grad():
            self.pos_emb.weight.add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE