MECHANISM: Position-embedding global-shift gauge fixing

HYPOTHESIS: An orthonormal zero-mean parameterization of the positional embedding will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because adding the same scalar to every positional-embedding coordinate only adds an all-ones residual-stream shift erased by every LayerNorm.

INTENDED_EDIT: Replace the positional embedding with a learned orthonormal coordinate parameterization spanning the zero-mean subspace, removing exactly one redundant scalar while preserving the quotient initialization distribution.

EVIDENCE: The first-LayerNorm gauge fixing reached 99.92% with 1607 parameters, while reductions that removed query capacity or projection weights failed; this targets another exact normalization symmetry without reducing attention or MLP capacity.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class ShiftGaugeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        total = num_embeddings * embedding_dim
        self.coordinates = nn.Parameter(torch.empty(total - 1))
        nn.init.normal_(self.coordinates, mean=0.0, std=0.02)

        direction = torch.full((total,), -1.0 / math.sqrt(total))
        direction[-1] += 1.0
        direction = direction / direction.norm()
        self.register_buffer("gauge_direction", direction, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = F.pad(self.coordinates, (0, 1))
        flat = flat - 2.0 * self.gauge_direction * torch.dot(self.gauge_direction, flat)
        weight = flat.view(self.num_embeddings, self.embedding_dim)
        return F.embedding(indices, weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = ShiftGaugeEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE