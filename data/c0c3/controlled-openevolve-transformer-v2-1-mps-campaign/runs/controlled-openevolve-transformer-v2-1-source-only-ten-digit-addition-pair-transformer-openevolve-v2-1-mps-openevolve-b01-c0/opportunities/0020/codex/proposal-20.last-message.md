MECHANISM: Tokenwise LayerNorm-null positional gauge

HYPOTHESIS: Centering the first positional-embedding row will reduce the model from 1629 to 1628 parameters while retaining at least 99% accuracy, because its removed featurewise common offset is invisible to every tokenwise LayerNorm and all seven effective contrasts remain learned.

INTENDED_EDIT: Replace the first positional-embedding row with seven centered learned contrasts, retain all other rows unchanged, and preserve the original initialization RNG sequence.

EVIDENCE: A third centered `fc2` column collapsed to 52.22%, while balanced bias gauges in `ln1`, `ln2`, and terminal `fc2` passed; this motivates testing the same distributed one-parameter gauge in a distinct exact LayerNorm-null pathway without further perturbing MLP hidden-unit optimization.

<<<<<<< SEARCH
        return self.drop(F.linear(hidden, weight, output_bias))


class CenteredBiasLayerNorm(nn.Module):
=======
        return self.drop(F.linear(hidden, weight, output_bias))


class CenteredFirstPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first_row = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        final_coordinate = -self.first_row.sum().reshape(1)
        first_row = torch.cat((self.first_row, final_coordinate))
        weight = torch.cat((first_row.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)


class CenteredBiasLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = CenteredFirstPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, MLP):
=======
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, CenteredFirstPositionEmbedding):
            full_weight = module.first_row.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            first_row = full_weight[0] - full_weight[0].mean()
            with torch.no_grad():
                module.first_row.copy_(first_row[:-1])
                module.rest.copy_(full_weight[1:])
        elif isinstance(module, MLP):
>>>>>>> REPLACE