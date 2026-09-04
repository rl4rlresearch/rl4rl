MECHANISM: RNG-aligned token–position translation gauge

HYPOTHESIS: Anchoring the penultimate feature of the final token embedding will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because a shared token-feature translation can be canceled by the opposite positional-feature translation without changing inputs or output probabilities.

INTENDED_EDIT: Store two fewer token-embedding scalars, reconstruct both as zero, and canonicalize initialization by shifting the removed feature across all token and positional embeddings while preserving the verified RNG stream and initial model function.

EVIDENCE: The current 1,576-parameter model achieves 99.92% with one final-token coordinate already anchored; this extends the embedding gauge through an exact joint token–position symmetry rather than adding another position-specific constraint like the failed adjacent positional-row reduction.

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
    """Tied token embedding with common-mode and token-position gauges anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-2].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 2)).view(
            self.num_embeddings, self.embedding_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OutputAnchoredLinear):
=======
        self._token_position_shift = None
        self.apply(self._init_weights)
        del self._token_position_shift

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, TokenAnchoredEmbedding):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            full_weight = F.pad(baseline_weight, (0, 1)).view(
                module.num_embeddings, module.embedding_dim
            )
            anchor = full_weight[-1, -2].clone()
            full_weight[:, -2].sub_(anchor)
            self._token_position_shift = anchor
            with torch.no_grad():
                module.weight.copy_(full_weight.flatten()[:-2])
        elif isinstance(module, OutputAnchoredLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
=======
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            full_weight[:, -2].add_(self._token_position_shift)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
>>>>>>> REPLACE