MECHANISM: Dual-orientation token–position embedding gauge

HYPOTHESIS: Centering the fourth positional-embedding feature column and transferring its mean into the corresponding token-embedding column will reduce the model from 1626 to 1625 parameters while retaining at least 99% accuracy, because transformer inputs remain exact and tied logits change only by a class-independent offset.

INTENDED_EDIT: Keep the first three token-centered gauges, but remove the fourth joint redundancy by representing positional feature four with centered contrasts and transferring its initialization mean to every token embedding.

EVIDENCE: Three token-centered joint gauges passed at 99.28%, while applying the same token-side parameterization to a fourth feature collapsed to 31.73%; orienting the fourth exact gauge toward the positional embedding tests whether that failure was caused by the token-side optimization geometry.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())


class Block(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())


class CenteredPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Consume the same constructor RNG as the original embedding.
        _ = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fourth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))
        self.register_buffer(
            "fourth_column_shift", torch.zeros(()), persistent=False
        )

    def _set_from_full(self, weight: torch.Tensor) -> None:
        fourth_column = weight[:, 3]
        with torch.no_grad():
            self.fourth_column.copy_(
                fourth_column[:-1] - fourth_column[-1]
            )
            self.rest.copy_(
                torch.cat((weight[:, :3], weight[:, 4:]), dim=1)
            )
            self.fourth_column_shift.copy_(fourth_column.mean())

    def tied_weight(self) -> torch.Tensor:
        anchored = torch.cat(
            (self.fourth_column, self.fourth_column.new_zeros(1))
        )
        fourth_column = anchored - anchored.mean()
        return torch.cat(
            (
                self.rest[:, :3],
                fourth_column.unsqueeze(1),
                self.rest[:, 3:],
            ),
            dim=1,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = CenteredTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = CenteredTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = CenteredPositionalEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
        token_shift = full_token_weight[:, :3].mean(dim=0)
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.weight[:, :3].add_(token_shift)
=======
        nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
        token_shift = full_token_weight[:, :3].mean(dim=0)
        full_token_weight[:, 3].add_(self.pos_emb.fourth_column_shift)
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.rest[:, :3].add_(token_shift)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, CenteredTokenEmbedding):
            full_weight = module.first_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, MLP):
=======
        elif isinstance(module, CenteredTokenEmbedding):
            full_weight = module.first_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, CenteredPositionalEmbedding):
            full_weight = module.fourth_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, MLP):
>>>>>>> REPLACE