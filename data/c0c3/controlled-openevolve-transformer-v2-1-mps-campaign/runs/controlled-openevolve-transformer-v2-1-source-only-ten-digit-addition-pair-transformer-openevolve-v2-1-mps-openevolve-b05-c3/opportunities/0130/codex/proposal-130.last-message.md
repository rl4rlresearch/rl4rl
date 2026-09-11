MECHANISM: Six-coordinate learned token bottleneck

HYPOTHESIS: Constraining the tied token codebook to six learned coordinates while retaining the eight-dimensional attention/MLP workspace will reduce the 1,123-parameter model by `VOCAB_SIZE` parameters and still achieve at least 99% accuracy.

INTENDED_EDIT: Fix a second embedding coordinate to zero, initialize the reduced codebook from the existing distribution, and optimize it directly while leaving the learned attention, 12-unit nonlinear bank, biases, decoding, and checkpoint flow unchanged.

EVIDENCE: The current internal computation reaches 99.96%, whereas reducing the 12-unit MLP collapsed to 0.01% and constraining attention or residual biases also failed. This challenges the untested assumption that the token input/output interface needs all seven quotient coordinates without disturbing those demonstrated load-bearing components.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Embedding):
    """Tied embedding modulo one hidden-coordinate shift per token."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.num_embeddings, 1),
            ),
            dim=-1,
        )
=======
class GaugeFixedEmbedding(nn.Embedding):
    """Tied learned codebook restricted to six hidden coordinates."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 2)
        )

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.num_embeddings, 2),
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-1])
=======
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-2])
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                embedding.weight.copy_(full[:, :-1])
=======
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                embedding.weight.copy_(full[:, :-2])
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_gauge_parameters = [
        model.token_emb.weight
    ]
=======
    embedding_gauge_parameters = []
>>>>>>> REPLACE