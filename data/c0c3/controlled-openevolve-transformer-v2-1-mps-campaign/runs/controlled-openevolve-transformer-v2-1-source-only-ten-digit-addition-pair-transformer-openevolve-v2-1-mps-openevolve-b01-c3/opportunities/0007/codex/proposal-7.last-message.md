MECHANISM: Mean-zero positional gauge fixing

HYPOTHESIS: Constraining positional embeddings to the seven-dimensional mean-zero subspace, alongside the verified affine-free block norms, will reduce parameters below 1,596 while retaining at least 99% accuracy because position-dependent all-ones components are removed by every downstream LayerNorm and never affect logits.

INTENDED_EDIT: Use an orthonormally parameterized mean-zero positional embedding and disable affine parameters in both block LayerNorms, while preserving the verified attention and MLP capacities.

EVIDENCE: Affine-free block norms achieved 99.81% accuracy with 1,596 parameters; the failed 1,560-parameter projection rewrite motivates leaving QKV and MLP projections unchanged and instead removing only the position-embedding direction that is exactly invisible to the pre-norm residual network and final LayerNorm.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        centered = torch.eye(embedding_dim) - torch.full(
            (embedding_dim, embedding_dim), 1.0 / embedding_dim
        )
        basis = torch.linalg.qr(centered[:, :-1], mode="reduced").Q
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE