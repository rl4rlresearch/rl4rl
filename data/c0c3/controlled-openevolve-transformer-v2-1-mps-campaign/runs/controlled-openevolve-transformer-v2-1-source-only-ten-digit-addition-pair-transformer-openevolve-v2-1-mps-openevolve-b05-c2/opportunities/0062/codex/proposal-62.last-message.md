MECHANISM: Complete anchor-token/position common-shift gauge

HYPOTHESIS: Fixing the seventh and final mean-zero coordinate of the anchor token will produce a 1516-parameter model with at least 99% accuracy while preserving initialized inputs and output softmax probabilities.

INTENDED_EDIT: Extend the qualified six-coordinate embedding gauge to all seven mean-zero coordinates, remove the now-empty anchor parameter entirely, reconstruct the anchor row as zero, and compensate the full shift in positional embeddings.

EVIDENCE: The fifth gauge achieved 100% accuracy at 1518 parameters and the sixth achieved 99.99% at 1517; completing the same exact symmetry is the closest supported reduction and avoids the value-rotation directions that previously harmed accuracy.

<<<<<<< SEARCH
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
        shift = last_coords[-4:]
        gauged_weight = gauged_weight - self.basis[:, -4:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-4])
            self.initial_position_shift.copy_(shift)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 4))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
=======
    @property
    def weight(self) -> torch.Tensor:
        last_row = self.weight_rows.new_zeros(self.embedding_dim)
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.pos_emb.weight[:, -4:].add_(
                self.token_emb.initial_position_shift
            )
=======
        with torch.no_grad():
            self.pos_emb.weight.add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE