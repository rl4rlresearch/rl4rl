MECHANISM: Joint token–position embedding gauge

HYPOTHESIS: Constraining one token-embedding feature column to centered contrasts will produce a 1628-parameter model with at least 99% accuracy, because shifting the removed column mean into every positional embedding preserves all transformer inputs exactly, while the tied output logits change only by a class-independent offset.

INTENDED_EDIT: Replace one token-embedding column with balanced learned contrasts, transfer its initialization mean to the matching positional column, and reconstruct the tied output weight dynamically while preserving the original RNG sequence.

EVIDENCE: The isolated positional-row gauge failed at 64.11%, whereas balanced terminal contrast gauges passed through 1629 parameters; this patch instead removes an exact joint token/position/output redundancy without perturbing the represented initial function.

<<<<<<< SEARCH
class Block(nn.Module):
=======
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


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = CenteredTokenEmbedding(cfg.vocab_size, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        # Preserve the original output-layer constructor RNG; its weight is
        # reconstructed from the centered input embedding during forward.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = None

        self.apply(self._init_weights)

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, MLP):
=======
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, CenteredTokenEmbedding):
            full_weight = module.first_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, MLP):
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, self.token_emb.tied_weight())
>>>>>>> REPLACE