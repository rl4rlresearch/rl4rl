MECHANISM: Second token–position embedding translation gauge

HYPOTHESIS: Fixing a second positional-embedding origin coordinate will reduce the verified 1,617-parameter model to 1,616 parameters while retaining at least 99% accuracy, because an equal coordinate-wise translation can be absorbed exactly into every token embedding without changing initialized hidden inputs.

INTENDED_EDIT: Absorb `pos_emb[0, 1]` into token-embedding coordinate 1, then omit the first two flattened positional parameters and reconstruct them as zeros.

EVIDENCE: The current model achieves 99.84% at 1,617 parameters while already using the identical gauge for coordinate 0; unlike the failed second MLP-row quotient at 98.38%, this extends a qualified embedding reparameterization without constraining a learned computational pathway.

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
            for coordinate in range(2):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
>>>>>>> REPLACE