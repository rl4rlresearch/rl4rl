MECHANISM: Residual second-head query-bias gauge fixing

HYPOTHESIS: Completing the proven embedding gauge and fixing one query-bias coordinate through the remaining second-head query-key rotation will produce a 1515-parameter model with at least 99% accuracy.

INTENDED_EDIT: Remove the anchor token’s remaining four coordinates, compensate all seven positional coordinates, and omit `q_bias[4]`, reconstructed as zero within the residual rows-4/5 query-key gauge.

EVIDENCE: The complete embedding gauge achieved 99.79% at 1516 parameters. Unlike the failed twelfth weight-based query-key constraint, the proposed residual gauge fixes an initially zero bias coordinate while preserving the initialized function.

<<<<<<< SEARCH
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

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 3))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
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
        with torch.no_grad():
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift
            )
=======
        with torch.no_grad():
            self.pos_emb.weight.add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE