MECHANISM: Two-coordinate token-position translation gauge fixing

HYPOTHESIS: Extending the verified embedding translation quotient to a second hidden coordinate will reduce the model to 1,628 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Transfer the first two positional coordinates into the tied token embeddings and reconstruct both corresponding positional scalars as zero.

EVIDENCE: The one-coordinate quotient achieved 99.95% accuracy at 1,629 parameters; the second coordinate has the same exact invariance, preserving initialized hidden inputs while changing logits only by a class-independent shift.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with one token-position translation gauge fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[1:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(1), self.weight))
        return F.embedding(
            idx,
            full_weight.view(self.num_embeddings, self.embedding_dim),
        )
=======
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with two token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[2:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(2), self.weight))
        return F.embedding(
            idx,
            full_weight.view(self.num_embeddings, self.embedding_dim),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
=======
        # Fix two exact token/position translation gauges while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, :2].detach().clone()
            self.token_emb.weight[:, :2].add_(offset)
            self.pos_emb.weight[:, :2].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
>>>>>>> REPLACE