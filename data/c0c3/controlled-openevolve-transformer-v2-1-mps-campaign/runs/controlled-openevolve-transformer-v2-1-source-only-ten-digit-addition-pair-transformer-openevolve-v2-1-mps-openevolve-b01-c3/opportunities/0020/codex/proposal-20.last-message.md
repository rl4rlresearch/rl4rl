MECHANISM: Four-dimensional positional routing codes

HYPOTHESIS: Replacing seven-dimensional positional embeddings with learned four-dimensional codes will reduce the model from 1,532 to 1,463 parameters while retaining at least 99% accuracy, because position primarily controls attention routing and the evidence identifies four—not seven—as the load-bearing routing width.

INTENDED_EDIT: Constrain all learned positional representations to a shared fixed four-dimensional mean-zero residual subspace while preserving full-width values, `d_ff=12`, two attention heads, and all verified gauge reductions.

EVIDENCE: The full model reached 99.94%, while reducing each head’s query/key width from four to three reached 98.18%; this motivates preserving four routing coordinates but challenges the shared assumption that positional representations require all seven LayerNorm-visible residual dimensions.

<<<<<<< SEARCH
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)
=======
class RoutingPositionEmbedding(nn.Embedding):
    """Positions represented in the attention head's routing dimension."""

    def __init__(self, num_embeddings: int, embedding_dim: int, routing_dim: int):
        super().__init__(num_embeddings, routing_dim)
        basis = mean_zero_basis(embedding_dim)[:, :routing_dim]
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = RoutingPositionEmbedding(
            cfg.max_seq_len, cfg.d_model, cfg.d_model // cfg.n_head
        )
>>>>>>> REPLACE