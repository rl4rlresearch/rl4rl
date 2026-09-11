MECHANISM: Second joint token–position embedding gauge

HYPOTHESIS: Centering a second token-embedding feature column will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because transferring its removed mean into the corresponding positional feature preserves transformer inputs exactly and changes tied logits only by a class-independent offset.

INTENDED_EDIT: Represent the first two token-embedding columns with balanced learned contrasts, transfer both initialization means into positional embeddings, and dynamically reconstruct the tied weight while preserving the original RNG sequence.

EVIDENCE: Centering the first token-embedding column with this exact joint gauge achieved 99.93% accuracy at 1628 parameters; applying the identical proven redundancy to one additional feature is the most conservative next reduction.

<<<<<<< SEARCH
class CenteredTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Consume the same constructor RNG as the original embedding.
        _ = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        column = weight[:, 0]
        with torch.no_grad():
            self.first_column.copy_(column[:-1] - column[-1])
            self.rest.copy_(weight[:, 1:])

    def tied_weight(self) -> torch.Tensor:
        anchored = torch.cat((self.first_column, self.first_column.new_zeros(1)))
        first_column = anchored - anchored.mean()
        return torch.cat((first_column.unsqueeze(1), self.rest), dim=1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())
=======
class CenteredTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Consume the same constructor RNG as the original embedding.
        _ = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.second_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 2))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_column = weight[:, 0]
        second_column = weight[:, 1]
        with torch.no_grad():
            self.first_column.copy_(first_column[:-1] - first_column[-1])
            self.second_column.copy_(second_column[:-1] - second_column[-1])
            self.rest.copy_(weight[:, 2:])

    @staticmethod
    def _centered_column(column: torch.Tensor) -> torch.Tensor:
        anchored = torch.cat((column, column.new_zeros(1)))
        return anchored - anchored.mean()

    def tied_weight(self) -> torch.Tensor:
        first_column = self._centered_column(self.first_column)
        second_column = self._centered_column(self.second_column)
        return torch.cat(
            (first_column.unsqueeze(1), second_column.unsqueeze(1), self.rest),
            dim=1,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The original tied output layer reinitialized the shared embedding at
        # the end of apply(). Reproduce that draw, then move the eliminated
        # common token-column offset into every positional embedding.
        full_token_weight = self.token_emb.first_column.new_empty(
            cfg.vocab_size, cfg.d_model
        )
        nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
        token_shift = full_token_weight[:, 0].mean()
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.weight[:, 0].add_(token_shift)
=======
        # The original tied output layer reinitialized the shared embedding at
        # the end of apply(). Reproduce that draw, then move the eliminated
        # common token-column offsets into every positional embedding.
        full_token_weight = self.token_emb.first_column.new_empty(
            cfg.vocab_size, cfg.d_model
        )
        nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
        token_shift = full_token_weight[:, :2].mean(dim=0)
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.weight[:, :2].add_(token_shift)
>>>>>>> REPLACE