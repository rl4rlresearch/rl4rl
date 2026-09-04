MECHANISM: Sixth token–position embedding translation gauge

HYPOTHESIS: Fixing positional-origin coordinate 5 will reduce the verified model to 1,612 parameters while retaining at least 99% accuracy, because the token–position translation preserves hidden inputs and changes tied-head logits only by a vocabulary-uniform shift.

INTENDED_EDIT: Absorb `pos_emb[0, 5]` into token-embedding coordinate 5, then omit the first six flattened positional parameters and reconstruct them as zeros.

EVIDENCE: The same exact gauge succeeded for coordinate 4 at 1,613 parameters with 99.94% accuracy, following successful applications at coordinates 1–3.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with five token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[5:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(5), self.weight))
=======
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with six token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[6:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(6), self.weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five exact token/position translation gauges while preserving
        # the initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(5):
=======
        # Fix six exact token/position translation gauges while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(6):
>>>>>>> REPLACE