MECHANISM: Vocabulary-balanced fourth token/position shift gauge

HYPOTHESIS: Centering one additional embedding coordinate across vocabulary, while retaining the qualified three final-row anchors, will yield 1576 parameters and at least 99% accuracy because the orthonormal balanced parameterization avoids concentrating four constraints on one token.

INTENDED_EDIT: Extend to three final-row gauges and remove a fourth shift degree of freedom by representing its vocabulary values in a mean-zero basis, with exact positional compensation at initialization.

EVIDENCE: Three final-row anchors achieved 99.8% at 1577 parameters, while fourth anchors concentrated on the final row reached only 95.9% and 89.18%; this tests whether balanced gauge conditioning, rather than the fourth exact gauge itself, caused those failures.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and two token/position shift gauges fixed."""
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
        shift = last_coords[-2:]
        gauged_weight = gauged_weight - basis[:, -2:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-2])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-2:]
        gauged_weight = gauged_weight - self.basis[:, -2:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-2])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 2))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with three final-row and one balanced shift gauges fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        vocab_basis = _mean_zero_basis(num_embeddings)
        common = torch.ones(embedding_dim, 1) / math.sqrt(embedding_dim)
        row_basis = torch.cat(
            (common, basis[:, :-4], basis[:, -3:]), dim=1
        )
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("vocab_basis", vocab_basis, persistent=False)
        self.register_buffer("row_basis", row_basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor

        last_coords = gauged_weight[-1] @ basis
        tail_shift = last_coords[-3:]
        gauged_weight = gauged_weight - basis[:, -3:] @ tail_shift

        balanced_values = gauged_weight @ basis[:, -4]
        balanced_shift = balanced_values.mean()
        gauged_weight = gauged_weight - basis[:, -4] * balanced_shift
        balanced_values = balanced_values - balanced_shift
        residual = gauged_weight - (
            balanced_values.unsqueeze(1) * basis[:, -4].unsqueeze(0)
        )

        self.weight_rows = nn.Parameter(residual[:-1] @ row_basis)
        self.gauge_weight = nn.Parameter(
            vocab_basis.transpose(0, 1) @ balanced_values
        )
        self.last_weight = nn.Parameter(residual[-1] @ basis[:, :-4])

        position_shift = torch.zeros_like(last_coords)
        position_shift[-4] = balanced_shift
        position_shift[-3:] = tail_shift
        self.register_buffer(
            "initial_position_shift",
            position_shift.detach().clone(),
            persistent=False,
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor

        last_coords = gauged_weight[-1] @ self.basis
        tail_shift = last_coords[-3:]
        gauged_weight = gauged_weight - self.basis[:, -3:] @ tail_shift

        balanced_values = gauged_weight @ self.basis[:, -4]
        balanced_shift = balanced_values.mean()
        gauged_weight = gauged_weight - self.basis[:, -4] * balanced_shift
        balanced_values = balanced_values - balanced_shift
        residual = gauged_weight - (
            balanced_values.unsqueeze(1) * self.basis[:, -4].unsqueeze(0)
        )

        position_shift = torch.zeros_like(last_coords)
        position_shift[-4] = balanced_shift
        position_shift[-3:] = tail_shift
        with torch.no_grad():
            self.weight_rows.copy_(residual[:-1] @ self.row_basis)
            self.gauge_weight.copy_(
                self.vocab_basis.transpose(0, 1) @ balanced_values
            )
            self.last_weight.copy_(residual[-1] @ self.basis[:, :-4])
            self.initial_position_shift.copy_(position_shift)

    @property
    def weight(self) -> torch.Tensor:
        rows = self.weight_rows @ self.row_basis.transpose(0, 1)
        last_row = self.basis[:, :-4] @ self.last_weight
        base = torch.cat((rows, last_row.unsqueeze(0)), dim=0)
        balanced_values = self.vocab_basis @ self.gauge_weight
        balanced = (
            balanced_values.unsqueeze(1) * self.basis[:, -4].unsqueeze(0)
        )
        return base + balanced

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized input function after fixing the additional
        # token/position shift gauge.
        with torch.no_grad():
            self.pos_emb.weight[:, -2:].add_(
                self.token_emb.initial_position_shift
            )
=======
        # Preserve the initialized input function after fixing the additional
        # token/position shift gauges.
        with torch.no_grad():
            self.pos_emb.weight.add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE