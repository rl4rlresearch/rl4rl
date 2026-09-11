MECHANISM: RNG-aligned second reference-token embedding anchor

HYPOTHESIS: Anchoring a second reference-token embedding coordinate while preserving every retained initialization value and the verified 1,580-parameter design’s downstream RNG state will achieve at least 99% accuracy with 1,579 parameters.

INTENDED_EDIT: Store two fewer token-embedding scalars, reconstruct both as zeros, and initialize from a baseline draw matching the current one-anchor model before discarding the newly anchored scalar.

EVIDENCE: The direct second token-coordinate anchor reached 96.04%, while RNG-aligning the previously failed third `fc2` constraint improved accuracy from 73.46% to 99.89%; this tests whether initialization and downstream-RNG disruption caused the near-threshold token-anchor failure.

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
    """Tied token embedding with two reference coordinates anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-2].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 2)).view(
            self.num_embeddings, self.embedding_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, PositionAnchoredEmbedding):
=======
        elif isinstance(module, TokenAnchoredEmbedding):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(baseline_weight[:-1])
        elif isinstance(module, PositionAnchoredEmbedding):
>>>>>>> REPLACE