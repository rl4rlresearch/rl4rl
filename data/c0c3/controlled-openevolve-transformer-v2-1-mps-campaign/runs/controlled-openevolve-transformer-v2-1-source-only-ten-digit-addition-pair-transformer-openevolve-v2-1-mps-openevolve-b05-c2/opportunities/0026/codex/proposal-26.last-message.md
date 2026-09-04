MECHANISM: Pre-MLP LayerNorm scale folding

HYPOTHESIS: Combining the qualified 1563-parameter design with an affine-free `ln2` will produce a 1555-parameter model with at least 99% accuracy, because `ln2`’s learned scale can be absorbed exactly into the unconstrained input columns of `fc1`.

INTENDED_EDIT: Apply the qualified three-coordinate token/position gauge and bias-free `ln1`, then remove the eight redundant `ln2` scale parameters.

EVIDENCE: The 1563-parameter design achieved 99.94% accuracy; its `ln2` is already bias-free, and removing its remaining scale preserves model expressivity and the identity-scale initialization through the following learned linear layer.

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
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and three token/position gauges fixed."""
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
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - basis[:, -3:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-3])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(gauged_weight[-1] @ self.basis)
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - self.basis[:, -3:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-3])
            self.initial_position_shift.copy_(shift)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ self.last_weight
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
=======
    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 3))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        with torch.no_grad():
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift
            )

    @staticmethod
>>>>>>> REPLACE