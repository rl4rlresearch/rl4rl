MECHANISM: Second tied-embedding translation gauge

HYPOTHESIS: Anchoring the reference token’s second-to-last embedding coordinate will reduce the model from 1,575 to 1,574 parameters while retaining at least 99% accuracy, because a feature-wise offset shared across token embeddings can be absorbed by positional embeddings and changes output logits only by a softmax-invariant common shift.

INTENDED_EDIT: Fix the final two flattened token-embedding coordinates at zero instead of only the final coordinate, while consuming the same 119 initialization draws as the verified design so all retained parameters and downstream initialization remain RNG-aligned.

EVIDENCE: The 99.91%-accurate 1,575-parameter design already succeeds with one axis-aligned reference-token embedding anchor. Recent 1,574-parameter failures altered the optimizer geometry of attention or MLP output columns, so extending the existing embedding gauge while leaving retained coordinates unchanged is a distinct and better-isolated reduction.

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
    """Tied token embedding with two feature-wise translation gauges anchored."""

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
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
=======
        elif isinstance(module, TokenAnchoredEmbedding):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(baseline_weight[:-1])
        elif isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
>>>>>>> REPLACE