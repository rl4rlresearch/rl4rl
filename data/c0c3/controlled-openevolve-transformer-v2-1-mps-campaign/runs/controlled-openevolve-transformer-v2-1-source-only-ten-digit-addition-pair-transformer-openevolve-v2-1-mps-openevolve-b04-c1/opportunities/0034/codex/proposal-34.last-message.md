MECHANISM: Single-position residual-shift quotient

HYPOTHESIS: Constraining only the first positional-embedding row to an isotropic seven-dimensional zero-mean basis will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because its omitted all-ones shift is erased by every LayerNorm without coupling optimization across positions.

INTENDED_EDIT: Replace one dense positional row with a `ShiftGaugeBias` row while leaving every other positional row independently dense and normally initialized.

EVIDENCE: Quotienting all positional rows failed at 55.57%, and the global positional quotient failed at 49.92%, but both disturbed the entire table; the same eight-to-seven isotropic quotient succeeded for the MLP output bias at 99.92%, motivating a localized one-row test.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class SingleRowShiftGaugeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.first_row = ShiftGaugeBias(embedding_dim)
        self.remaining_rows = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.first_row.coordinates, mean=0.0, std=0.02)
        nn.init.normal_(self.remaining_rows, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.first_row().unsqueeze(0), self.remaining_rows), dim=0)
        return F.embedding(idx, weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = SingleRowShiftGaugeEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
=======
        if isinstance(module, (ShiftGaugeEmbedding, SingleRowShiftGaugeEmbedding)):
            module.reset_parameters()
>>>>>>> REPLACE