MECHANISM: Joint token–position translation gauge

HYPOTHESIS: Anchoring a second coordinate of the reference token embedding will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because translating that coordinate for every token can be absorbed by the positional embeddings, while its output-logit effect is a class-independent softmax shift.

INTENDED_EDIT: Store two fewer token-embedding scalars and reconstruct the final two coordinates as zeros.

EVIDENCE: The current design achieves 99.92% accuracy with one reference-token coordinate already anchored; this tests an independent exact token–position gauge after the adjacent third `ln1` bias and ninth positional-row constraints failed.

<<<<<<< SEARCH
class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with one global common-mode scalar anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )
=======
class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with two reference-token coordinates anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-2].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 2)).view(
            self.num_embeddings, self.embedding_dim
        )
>>>>>>> REPLACE